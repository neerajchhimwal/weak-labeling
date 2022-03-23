import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import os

from labeling_functions import *
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis
from snorkel.labeling import filter_unlabeled_dataframe

from config import POSITIVE, NEGATIVE, NEUTRAL

def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return NEGATIVE
    elif rating == 3:
        return NEUTRAL
    else:
        return POSITIVE

def get_splits(df, train_ratio, valid_ratio, test_ratio):
    '''
    applies sklearn train_test split twice:
        - first splits between train+valid and test
        - then splits train+valid into train and valid
    '''
    x, x_test, y, y_test = train_test_split(df['clean_title'], df['sentiment'], test_size=test_ratio, train_size=train_ratio+valid_ratio)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=valid_ratio, train_size=1-valid_ratio)
    return pd.DataFrame(x_train), pd.DataFrame(x_valid), pd.DataFrame(x_test), pd.DataFrame(y_train), pd.DataFrame(y_valid), pd.DataFrame(y_test)


def train_valid_test_split(df, train_ratio=0.8, valid_ratio=0.15, test_ratio=0.05):
    '''
    inputs:
        - df: dataframe with at least two columns: "clean_title" and "rating"
        - sentiment_id_to_label_dict: sentiment id to label map (e.g. {'NEGATIVE': 0, 'POSITIVE': 1, 'NEUTRAL': 2})
        - train, valid, test ratio: percentage reviews to use as train, vlaid, test respectively
    
    these splits are done from each sentiment category and then concatenated
    sentiment categories defined as:
        - positive: 4 and 5 star reviews
        - negative: <=2 star reviews
        - neutral: 3 star reviews
    returns:
        - train valid and test dfs, train valid and test label dfs
    '''

    # print(df.shape)

    df['sentiment'] = df['rating'].apply(to_sentiment)
    
    pos_df = df[df['sentiment']==POSITIVE].reset_index(drop=True)
    neg_df = df[df['sentiment']==NEGATIVE].reset_index(drop=True)
    neutral_df = df[df['sentiment']==NEUTRAL].reset_index(drop=True)    
    
    
    # print(pos_df.shape)
    # print(neg_df.shape)
    # print(neutral_df.shape)
    
    x_train_pos, x_valid_pos, x_test_pos, y_train_pos, y_valid_pos, y_test_pos = get_splits(pos_df, train_ratio, valid_ratio, test_ratio)
    x_train_neg, x_valid_neg, x_test_neg, y_train_neg, y_valid_neg, y_test_neg = get_splits(neg_df, train_ratio, valid_ratio, test_ratio)
    x_train_neu, x_valid_neu, x_test_neu, y_train_neu, y_valid_neu, y_test_neu = get_splits(neutral_df, train_ratio, valid_ratio, test_ratio)
    
    df_train = pd.concat([x_train_pos, x_train_neg, x_train_neu]).reset_index(drop=True)
    df_valid = pd.concat([x_valid_pos, x_valid_neg, x_valid_neu]).reset_index(drop=True)
    df_test = pd.concat([x_test_pos, x_test_neg, x_test_neu]).reset_index(drop=True)
    
    df_train_labels = pd.concat([y_train_pos, y_train_neg, y_train_neu]).reset_index(drop=True)
    df_valid_labels = pd.concat([y_valid_pos, y_valid_neg, y_valid_neu]).reset_index(drop=True)
    df_test_labels = pd.concat([y_test_pos, y_test_neg, y_test_neu]).reset_index(drop=True)

    
    return df_train, df_valid, df_test, df_train_labels, df_valid_labels, df_test_labels

def train_labeling_functions(df_train, df_valid, df_train_labels, df_valid_labels):
    final_lfs = [polarity_positive, polarity_negative, polarity_neutral,
            keyword_stopped, keyword_worst, regex_stop, keyword_ok, regex_good_quality, regex_value_for_money,
            sentiment_positive, sentiment_negative, sentiment_neutral_3_star, sentiment_neutral_from_positive]

    applier = PandasLFApplier(final_lfs)
    L_train_final = applier.apply(df_train)
    L_valid_final = applier.apply(df_valid)

    # saving LF analysis summary
    lf_analysis_df_valid = LFAnalysis(L_valid_final, final_lfs).lf_summary(df_valid_labels.sentiment.values)
    # lf_analysis_df_valid = LFAnalysis(L_valid_final, final_lfs).lf_summary()
    out_dir = './results'
    os.makedirs(out_dir, exist_ok=True)
    lf_analysis_df_valid.to_csv(os.path.join(out_dir, 'lf_analysis_df_valid.csv'), index=False)

    # training Label Model
    print('Training Label Model...')
    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(L_train=L_train_final, n_epochs=1000, log_freq=100, seed=123)

    # getting predictions
    print('Getting predictions from Label Model...')
    preds_train = label_model.predict(L_train_final)
    preds_dev = label_model.predict(L_valid_final)

    # removing lines for which LFs abstained (didn't vote)
    df_train_filtered, preds_train_filtered = filter_unlabeled_dataframe(
        X=df_train, y=preds_train, L=L_train_final
    )

    df_valid_filtered, preds_valid_filtered = filter_unlabeled_dataframe(
        X=df_valid, y=preds_dev, L=L_valid_final
    )

    df_train_filtered['preds'] = preds_train_filtered
    df_valid_filtered['preds'] = preds_valid_filtered


if __name__ == "__main__":
    clean_review_csv = '../data/boat_bassheads_25k_clean.csv'
    
    df = pd.read_csv(clean_review_csv)
    
    df_train, df_valid, df_test, df_train_labels, df_valid_labels, df_test_labels = train_valid_test_split(df)

    print('Sentiment dist in Train:')
    print(df_train_labels['sentiment'].value_counts(normalize=True))
    print('Sentiment dist in Valid:')
    print(df_valid_labels['sentiment'].value_counts(normalize=True))
    print('Sentiment dist in Test:')
    print(df_test_labels['sentiment'].value_counts(normalize=True))

    # Saving

