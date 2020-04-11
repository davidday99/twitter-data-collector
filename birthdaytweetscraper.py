import GetOldTweets3 as got
import pandas as pd

text_query = 'happy {}th birthday'
age = 18  # set age to search for
count = 500  # set number of results to fetch


tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query.format(age)).setMaxTweets(count)

tweets = got.manager.TweetManager.getTweets(tweetCriteria)

tweet_user = [tweet.username for tweet in tweets]
tweet_text = [tweet.text for tweet in tweets]


tweet_data = pd.DataFrame({'username': tweet_user, 'tweet': tweet_text})

tweet_data.to_csv('{}yo.csv'.format(age), index=False, encoding='utf-8')  # table of users and their birthday tweets
