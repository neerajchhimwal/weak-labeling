from snorkel.labeling import LabelingFunction
from snorkel.labeling import labeling_function
# from snorkel.labeling.model import MajorityLabelVoter
from snorkel.preprocess import preprocessor
from textblob import TextBlob
from transformers import pipeline

import json
import re
from config import ABSTAIN, POSITIVE, NEGATIVE, NEUTRAL

'''
Lableing functions expect df to have a field called "clean_title"
'''

classifier_bert = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")

'''
LFs based on nltk sentiment- textblob
'''

@preprocessor(memoize=True)
def textblob_polarity(x):
    scores = TextBlob(x.clean_title)
    x.polarity = scores.polarity
    return x

# Label high polarity as positive.
@labeling_function(pre=[textblob_polarity])
def polarity_positive(x):
    return POSITIVE if x.polarity > 0.3 else ABSTAIN

# Label low polarity as negative.
@labeling_function(pre=[textblob_polarity])
def polarity_negative(x):
    return NEGATIVE if x.polarity < -0.25 else ABSTAIN

@labeling_function(pre=[textblob_polarity])
def polarity_neutral(x):
    return NEUTRAL if x.polarity <= 0.3 and x.polarity >=0 else ABSTAIN

'''
LFs based on keywords and regex
'''

def keyword_lookup(x, keywords, label):
    if any(word in x.clean_title.lower() for word in keywords):
        return label
    return ABSTAIN

def make_keyword_lf(keywords, label):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )

# negative
keyword_stopped = make_keyword_lf(keywords=["stopped", "working"], label=NEGATIVE)
keyword_worst = make_keyword_lf(keywords=["worst"], label=NEGATIVE)

@labeling_function()
def regex_stop(x):
    return NEGATIVE if re.search(r".*not.*work", x.clean_title, flags=re.I) else ABSTAIN

# neutral
keyword_ok = make_keyword_lf(keywords=["ok"], label=NEUTRAL)

# positive
@labeling_function()
def regex_good_quality(x):
    return POSITIVE if re.search(r"good.*quality", x.clean_title, flags=re.I) else ABSTAIN

@labeling_function()
def regex_sound_quality(x):
    return POSITIVE if re.search(r"good.*sound.*quality", x.clean_title, flags=re.I) or re.search(r"best.*sound.*", x.clean_title, flags=re.I) or re.search(r"amazing.*sound", x.clean_title, flags=re.I) else ABSTAIN

@labeling_function()
def regex_value_for_money(x):
    return POSITIVE if re.search(r"value.*money", x.clean_title, flags=re.I) else ABSTAIN

'''
LFs based on BERT sentiment analysis
'''

@preprocessor(memoize=True)
def bert_sentiment(x):
    label = int(classifier_bert(x.clean_title)[0]['label'].split()[0])
    confidence = classifier_bert(x.clean_title)[0]['score']
    x.label = label
    x.confidence = confidence
    return x

# positive.
@labeling_function(pre=[bert_sentiment])
def sentiment_positive(x):
    return POSITIVE if x.label >= 4 and x.confidence > 0.6 else ABSTAIN

# negative.
@labeling_function(pre=[bert_sentiment])
def sentiment_negative(x):
    return NEGATIVE if x.label <= 2 and x.confidence > 0.6 else ABSTAIN

# neutral. 3 star with any confidence
@labeling_function(pre=[bert_sentiment])
def sentiment_neutral_3_star(x):
    return NEUTRAL if x.label == 3 else ABSTAIN

# neutral -> when the review isn't too positive
@labeling_function(pre=[bert_sentiment])
def sentiment_neutral_from_positive(x):
    return NEUTRAL if x.label >=4 and x.confidence < 0.5 else ABSTAIN