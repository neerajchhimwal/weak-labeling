import pandas as pd
import contractions
import emoji
from collections import Counter
from tqdm import tqdm
import re
import config 

def rating_to_sentiment():
    pass

def train_test_valid_split():
    pass

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

    '''


    df = pd.read_csv(csv_file)
    print('Dataframe Shape: ', df.shape)

    if df.title.isnull().sum() != 0:
        print('Removing lines with empty title field. Count = ', df.title.isnull().sum())
        df = df.dropna(subset = ["title"])
        print('Dataframe Shape: ', df.shape)

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
    
    print('Dataframe Shape: ', df.shape)

    all_chars_title = ''.join(list(df['clean_title']))
    char_count = Counter(list(all_chars_title))

    with open(out_dict, 'w+') as f:
        [print(f'{key} : {value}', file=f) for (key, value) in char_count.items()]

    print('Clean dictionary of title field text saved as: ', out_dict)

    df.to_csv(out_csv_file, index=False)
    print('DF with clean title field text saved as: ', out_csv_file)


if __name__ == "__main__":
    # amzn_reviews_file = ''
    # out_csv = amzn_reviews_file.replace('.csv', '_clean.csv')
    # out_dict = out_csv.replace('.csv', '.txt')

    clean(csv_file=config.amazon_review_csv_path, out_csv_file=config.clean_csv_out_path, out_dict=config.title_char_dict_clean)