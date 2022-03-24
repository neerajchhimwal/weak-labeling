import pandas as pd
import numpy as np
import os
import config
from labeling_functions import *
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis
from snorkel.labeling import filter_unlabeled_dataframe

def train_labeling_functions_and_predict(df_train, df_valid):
    final_lfs = [polarity_positive, polarity_negative, polarity_neutral,
            keyword_stopped, keyword_worst, regex_stop, keyword_ok, regex_good_quality, regex_sound_quality, regex_value_for_money,
            sentiment_positive, sentiment_negative, sentiment_neutral_3_star, sentiment_neutral_from_positive]

    applier = PandasLFApplier(final_lfs)
    L_train_final = applier.apply(df_train)
    L_valid_final = applier.apply(df_valid)

    # LF analysis summary
    # lf_analysis_df_valid = LFAnalysis(L_valid_final, final_lfs).lf_summary(df_valid['sentiment_from_rating'].values)
    lf_analysis_df_valid = LFAnalysis(L_valid_final, final_lfs).lf_summary()
    '''
    This LF analysis ideadlly should be logged in wandb as an artefact
    '''
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

    '''
    Saving train and valid files with predictions
    '''
    df_train_filtered.to_csv(os.path.join(config.out_training_csvs_dir, config.train_name_after_labeling), index=False)
    df_valid_filtered.to_csv(os.path.join(config.out_training_csvs_dir, config.valid_name_after_labeling), index=False)

    return df_train_filtered, df_valid_filtered

if __name__ == "__main__":
    df_train = pd.read_csv(config.train_csv_for_labeling)
    df_valid = pd.read_csv(config.valid_csv_for_labeling)
    # print(df_train.head())
    df_train_filtered, df_valid_filtered = train_labeling_functions_and_predict(df_train, df_valid)