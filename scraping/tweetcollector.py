import pandas as pd
import re
import GetOldTweets3 as got
import time

FIFTEEN_MIN_IN_SEC = 15 * 65  # add a little extra time to be safe

age = 18  # age used as a label for each data sample that is created, *THIS SHOULD MATCH age IN birthdaytweetscraper.py*
count = 10  # number of tweets to scrape from each user

df = pd.read_csv('{}yo.csv'.format(age))
users = []

tweets = df['tweet']

# get the username of the person who's being wished happy birthday, add to users list
for tweet in tweets:
    try:
        users.append(re.sub('[\W]', '',
                            tweet.split('@', 1)[1].split(' ')[0]))  # extract username, clean up, and add to list
    except IndexError:
        continue

users = list(set(users))  # remove repeated users

tweet_data = {}

print('Getting {} tweets from each of {} users.'.format(count, len(users), age))

userNum = 1

for user in users:
    # every 500 users, wait 15 minutes to avoid making too many HTTP requests and erroring out
    if (userNum % 500) == 0:
        print("*15 minute request cool-off*")
        startTime = time.time()
        currentTime = time.time()
        while currentTime - startTime < FIFTEEN_MIN_IN_SEC:
            currentTime = time.time()

    tweetCriteria = got.manager.TweetCriteria().setUsername(user).setMaxTweets(count)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    tweet_text = [tweet.text for tweet in tweets]

    # print(user + "has " + str(len(tweet_text)) + " tweets, adding "
    #      + str(count - len(tweet_text)) + " extra empty tweets.")
    # while len(tweet_text) < count:  # add empty elements if necessary to make every sample same length
    #    tweet_text.append('')

    if len(tweet_text) < count:  # don't use sample if user has less than 'count' tweets
        continue
    else:
        print('{}: Getting tweets from {}...'.format(userNum, user))
        userNum += 1

    tweet_data[user] = tweet_text

print('Done!')

print('{} samples created out of {} total users.'.format(len(tweet_data), len(users)))

tweet_ds = pd.DataFrame(tweet_data).transpose().reset_index()  # original dataframe has users as a row, so transpose

tweet_ds.rename(columns={'index': 'user'}, inplace=True)  # after transposing, need to add a name to user column

tweet_ds.insert(0, 'age', [age] * tweet_ds.shape[0], True)  # add age label

tweet_ds.to_csv('{}yo_dataset.csv'.format(age), index=False, encoding='utf-8')
