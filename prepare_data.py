'''
-> Takes scraped amazon reviews 
    -> cleans title field 
        -> prepares train, valid, test sets based on stratified splits on sentiment labels 
(at this point, sentiment labels are extracted from amazon ratings, for the purpose of sampling)
'''

import re
import config 
import os
import emoji
import pandas as pd
import numpy as np
import contractions

from collections import Counter
from tqdm import tqdm
from config import POSITIVE, NEGATIVE, NEUTRAL
from sklearn.model_selection import train_test_split

def remove_emojis_and_punc(text):
    text = str(text)
    emoji_free = emoji.get_emoji_regexp().sub(r'', text)
    
    # punct
    punc_and_emoji_free = re.sub('[%s]' % re.escape("!\"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~₹“”—…"), ' ', emoji_free)
    
    return ' '.join([word for word in punc_and_emoji_free.split() if word])

def clean(csv_file, out_csv_file, out_dict):
    '''
    removing empty lines (from "title" field only)
    removing emojis
    some lines have only emojis, removing them
    remove review in any lang other than en
    expand contractions (eg: "I've" -> "I have")

    returns: Pandas DF with new field "clean_title"

    '''


    df = pd.read_csv(csv_file)
    print('Dataframe Shape: ', df.shape)

    if df.title.isnull().sum() != 0:
        print('Removing lines with empty title field. Count = ', df.title.isnull().sum())
        df = df.dropna(subset = ["title"])
        print('Dataframe Shape: ', df.shape)
        df.reset_index(drop=True, inplace=True)

    print("Expanding contractions...")
    df['clean_title'] = [contractions.fix(text) for text in tqdm(df['title'])]

    print("Removing emojis and punc...")
    df['clean_title'] = df['clean_title'].apply(remove_emojis_and_punc)

    # these are titles where only an emoji was present and hence are now emplty after cleaning
    
    indices_to_remove = [index for index, i in enumerate(df.clean_title) if not i]

    print('Removing lines with foreign chars...')
    pattern = f"[^ A-Za-z0-9]+"
    for i, line in enumerate(tqdm(df['clean_title'])):
        if re.search(pattern, line):
            # print(line)
            indices_to_remove.append(i)

    # Removing lines with foreign chars in content may lead to many lines being removed

    indices_to_remove = list(set(indices_to_remove))
    print('Num of indices with chars in language other than En: ', len(indices_to_remove))

    if len(indices_to_remove) != 0:
        df = df.drop(indices_to_remove)
        df.reset_index(drop=True, inplace=True)

    if df.clean_title.isnull().sum() != 0:
        print('Removing lines with empty clean_title field. Count = ', df.clean_title.isnull().sum())
        df = df.dropna(subset = ["clean_title"])
        print('Dataframe Shape: ', df.shape)
        df.reset_index(drop=True, inplace=True)
    
    print('Dataframe Shape: ', df.shape)

    all_chars_title = ''.join(list(df['clean_title']))
    char_count = Counter(list(all_chars_title))

    '''
    Saving files: 
        out_dict: char count for clean_title field for analysis
        out_csv_file: clean DF with new field: "clean_title"
    '''
    with open(out_dict, 'w+') as f:
        [print(f'{key} : {value}', file=f) for (key, value) in char_count.items()]

    print('Clean dictionary of title field text saved as: ', out_dict)

    df.to_csv(out_csv_file, index=False)
    print('DF with clean title field text saved as: ', out_csv_file)

    return df

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
    x, x_test, y, y_test = train_test_split(df['clean_title'], df['sentiment_from_rating'], test_size=test_ratio, train_size=train_ratio+valid_ratio)
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

    df['sentiment_from_rating'] = df['rating'].apply(to_sentiment)
    
    pos_df = df[df['sentiment_from_rating']==POSITIVE].reset_index(drop=True)
    neg_df = df[df['sentiment_from_rating']==NEGATIVE].reset_index(drop=True)
    neutral_df = df[df['sentiment_from_rating']==NEUTRAL].reset_index(drop=True)    
    
    
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

    df_train['sentiment_from_rating'] = df_train_labels['sentiment_from_rating']
    df_valid['sentiment_from_rating'] = df_valid_labels['sentiment_from_rating']
    df_test['sentiment_from_rating'] = df_test_labels['sentiment_from_rating']

    '''
    Saving train, valid and test csv
    '''
    
    if df_train.clean_title.isnull().sum() != 0:
        df_train = df_train.dropna(subset = ["clean_title"])

    if df_valid.clean_title.isnull().sum() != 0:
        df_valid = df_valid.dropna(subset = ["clean_title"])
        
    df_train.to_csv(os.path.join(config.out_training_csvs_dir, 'train.csv'), index=False)
    df_valid.to_csv(os.path.join(config.out_training_csvs_dir, 'valid.csv'), index=False)
    df_test.to_csv(os.path.join(config.out_training_csvs_dir, 'test.csv'), index=False)

    print(f'Train, valid and test csvs saved at: {config.out_training_csvs_dir}')
    # return df_train, df_valid, df_test

if __name__ == "__main__":

    clean_df = clean(csv_file=config.amazon_review_csv_path, out_csv_file=config.clean_csv_out_path, out_dict=config.title_char_dict_clean)
    train_valid_test_split(clean_df)

