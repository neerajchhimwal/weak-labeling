import json
import os

'''
Labels mapping to be used
'''
id_to_label_json_path = './id_to_label.json'

# opening label mapping
with open(id_to_label_json_path) as f:
    sentiment_id_to_label_dict = json.load(f)

ABSTAIN = -1
POSITIVE = sentiment_id_to_label_dict['POSITIVE']
NEGATIVE = sentiment_id_to_label_dict['NEGATIVE']
NEUTRAL = sentiment_id_to_label_dict['NEUTRAL']

'''
Cleaning params
'''
amazon_review_csv_path = './data/raw/boat_bassheads_25k.csv'
clean_csv_out_path = './data/clean/boat_bassheads_25k_clean.csv'
title_char_dict_clean = './data/clean/boat_bassheads_25k_clean_dict.txt'

out_training_csvs_dir = './data/training/'

'''
For weak labeling
'''
train_csv_for_labeling = os.path.join(out_training_csvs_dir, 'train.csv')
valid_csv_for_labeling = os.path.join(out_training_csvs_dir, 'valid.csv')
train_name_after_labeling = 'train_with_preds.csv'
valid_name_after_labeling = 'valid_with_preds.csv'