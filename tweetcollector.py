import pandas as pd
import re
import GetOldTweets3 as got

df = pd.read_csv('18yo.csv')
age = 18  # age used as a label for each data sample that is created, *THIS SHOULD MATCH age IN birthdaytweetscraper.py*

users = []

tweets = df['tweet']

# get the username of the person who's being wished happy birthday, add to users list
for tweet in tweets:
    try:
        users.append(re.sub('[\W]', '',
                            tweet.split('@', 1)[1].split(' ')[0]))  # extract username, clean up, and add to list
    except IndexError:
        continue

tweet_data = {}

count = 10  # number of tweets to scrape from each user
for user in users:
    tweetCriteria = got.manager.TweetCriteria().setUsername(user).setMaxTweets(count)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    tweet_text = [tweet.text for tweet in tweets]

    print(user + "has " + str(len(tweet_text)) + " tweets, adding "
          + str(count - len(tweet_text)) + " extra empty tweets.")

    while len(tweet_text) < count:  # add empty elements if necessary to make every sample same length
        tweet_text.append('')

    tweet_data[user] = tweet_text

tweet_ds = pd.DataFrame(tweet_data).transpose().reset_index()  # original dataframe has users as a row, so transpose

tweet_ds.rename(columns={'index': 'user'}, inplace=True)  # after transposing, need to add a name to user column

tweet_ds.insert(0, 'age', [age] * tweet_ds.shape[0], True)  # add age label

tweet_ds.to_csv('dataset.csv', index=False, encoding='utf-8')
