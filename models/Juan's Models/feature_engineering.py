import nltk
import pandas as pd
import numpy as np
from better_profanity import profanity
import re
import enchant

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

train_df = pd.read_csv('data/train_w_logits.csv')
test_df = pd.read_csv('data/test_w_logits.csv')
data_all = pd.concat([train_df, test_df])

# Sentiment analysis as features
sid = SentimentIntensityAnalyzer()
data_all = data_all.assign(sentiment_pos=np.zeros(data_all.shape[0]))
data_all = data_all.assign(sentiment_neg=np.zeros(data_all.shape[0]))
data_all = data_all.assign(sentiment_neu=np.zeros(data_all.shape[0]))

for idx, row in data_all.iterrows():
    text = row['tweets_text']
    handle = row['handle']
    score = sid.polarity_scores(text)
    data_all.at[idx, 'sentiment_pos'] = score['pos']
    data_all.at[idx, 'sentiment_neg'] = score['neg']
    data_all.at[idx, 'sentiment_neu'] = score['neu']

# Punctuation as features
data_all = data_all.assign(num_periods=np.zeros(data_all.shape[0]))
data_all = data_all.assign(num_exclamation=np.zeros(data_all.shape[0]))
data_all = data_all.assign(num_questionmark=np.zeros(data_all.shape[0]))

for idx, row in data_all.iterrows():
    data_all.at[idx, 'num_periods'] = row['tweets_text'].count('.')
    data_all.at[idx, 'num_exclamation'] = row['tweets_text'].count('!')
    data_all.at[idx, 'num_questionmark'] = row['tweets_text'].count('?')

# Letter case as features
data_all = data_all.assign(pct_uppercase=np.zeros(data_all.shape[0]))
data_all = data_all.assign(pct_lowercase=np.zeros(data_all.shape[0]))
for idx, row in data_all.iterrows():
    num_upper = sum(1 for c in row['tweets_text'] if c.isupper())
    num_lower = sum(1 for c in row['tweets_text'] if c.islower())
    data_all.at[idx, 'pct_uppercase'] = num_upper/len(row['tweets_text'])
    data_all.at[idx, 'pct_lowercase'] = num_lower / len(row['tweets_text'])

# Contains profanity
data_all = data_all.assign(contains_profanity=np.zeros(data_all.shape[0]))
for idx, row in data_all.iterrows():
    if profanity.contains_profanity(row['tweets_text']):
        data_all.at[idx, 'contains_profanity'] = 1

internet_slang = ['omg', 'lmao', 'lol', 'ffs', 'lmfao', 'wtf', 'wth', 'fr', 'idk', 'idek', 'stfu', 'tbh', 'ily', 'smh',
                  'jk', 'af', 'imo', 'imho', 'idc', 'hmu', 'nvm', 'lmk', 'hbd', 'irl', 'rt', 'ofc']
data_all = data_all.assign(slang_count=np.zeros(data_all.shape[0]))
for idx, row in data_all.iterrows():
    slang_count = 0
    words = re.split(r'\W+', row['tweets_text'].lower())
    for word in words:
        if word in internet_slang:
            slang_count += 1
    data_all.at[idx, 'slang_count'] = slang_count

# Count properly spelled words
data_all = data_all.assign(pct_valid_words=np.zeros(data_all.shape[0]))
for idx, row in data_all.iterrows():
    words = re.split(r'\W+', row['tweets_text'].lower())
    d = enchant.Dict("en_US")
    correct_count = 0
    for word in words:
        if word == '':
            continue
        if d.check(word):
            correct_count += 1
    data_all.at[idx, 'pct_valid_words'] = correct_count / len(words)

train_df = data_all[:train_df.shape[0]]
test_df = data_all[train_df.shape[0]:]

print(train_df.head())
print(test_df.head())

test_df.to_csv('data/test_w_logits.csv', index=False)
train_df.to_csv('data/train_w_logits.csv', index=False)

# TODO: maybe add features based on repeated letters, uncommon words (unigrams, bigrams, trigrams)
