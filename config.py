import json

id_to_label_json_path = './id_to_label.json'

# opening label mapping
with open(id_to_label_json_path) as f:
    sentiment_id_to_label_dict = json.load(f)

ABSTAIN = -1
POSITIVE = sentiment_id_to_label_dict['POSITIVE']
NEGATIVE = sentiment_id_to_label_dict['NEGATIVE']
NEUTRAL = sentiment_id_to_label_dict['NEUTRAL']
