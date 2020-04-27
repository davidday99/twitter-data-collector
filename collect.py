#!/usr/bin/env python3
"""
collect.py
$ ./collect.py -a {age} -e {end_age_range} -n {number}
e.g.: $ ./collect.py -a 18 -n 20
      $ ./collect.py -a 18 -e 24 -n 20
      $ ./collect.py -a 18 -a 20 -e 24 -n 20
Note that the script automatically fills between the largest `-a` value
and the `-e` value (if passed)
Collects tweets and organizes them into datasets by age.
"""
# First party packages
import argparse
import os
import sys
import re
import time

# Third party packages
import pandas as pd
import GetOldTweets3 as got



def get_inputs():
    """
    Purpose:
        Gets inputs for collect.py
    Args:
        None
    Returns:
        args.age     (str): Start age
        args.end_age (str): End age if requiring age range
        args.count   (str): Number of tweets to get
    """
    # NOTE -e will add on ages starting from last age in `-a` list
    parser = argparse.ArgumentParser(description='Request tweets from GetOldTweets3 API')
    parser.add_argument('-a', '--age', dest='age', default=18, help='Age to get tweets for')
    parser.add_argument('-e', '--end', dest='end_age', default=24, help='End age to get tweets for')
    parser.add_argument('-n', '--number', dest='number', default=1, help='Number of tweets to get')
    args = parser.parse_args()

    return args.age, args.end_age, args.number



def find_ages(age, end_age):
    """
    Purpose:
        Produce list of ages to get from input params
    Args:
        age    (list): Start age string
        end_age  (str): Last age string to find data for
    Returns:
        age_list    (list): List of integer ages to find tweets for 
    """
    int_age = []

    age_diff = int(end_age) - int(age)
    if age_diff < 0:
        sys.exit("Last input age exceeds input age range")

    for i in range(int(age), int(end_age)+1):
        int_age.append(i)
    
    return int_age


def get_tweetset(age, count, data_path):
    """
    Purpose:
        Gets tweets for one age
    Args:
        age         (int): age used as a label for each data sample that is created,
                          *THIS SHOULD MATCH age IN birthdaytweetscraper.py*
        count       (int): number of tweets to scrape from each user
    Returns:
        None
    """
    FIFTEEN_MIN_IN_SEC = 15 * 65  # add a little extra time to be safe
    in_csv_name = '{}yo.csv'.format(age)
    if os.path.exists(data_path + in_csv_name) is False:
        print(in_csv_name + " not found! Downloading dataset...")
        get_age_tweets(age)
    
    df = pd.read_csv(in_csv_name)
    users = []

    tweets = df['tweet']

    # get the username of the person who's being wished happy birthday, add to users list
    suffix = "th"
    if age % 10 == 1:
    	suffix = 'st'
    elif age % 10 == 2:
    	suffix = 'nd'
    elif age % 10 == 3:
    	suffix = 'rd'

    for tweet in tweets:
        try:
        	# Add tweet only if it's "@user happy nth birthday" or "happy nth birthday @user"
        	query = 'happy ' + str(age) + suffix + ' birthday'
        	if query in tweet.lower():
        		users.append(re.sub('[\W]', '', tweet.split('birthday @', 1)[1].split(' ')[0]))  # extract username, clean up, and add to list
        		#print(tweet)
        	else:
        		split = tweet.split(' ')
        		prev_word = ''
        		for word in split:
        			if '@' in prev_word and word == 'happy' and query in tweet:
        				users.append(prev_word[1:])
        				#print(tweet)
        				break
        			prev_word = word
        except IndexError:
            continue
        except AttributeError:
            continue

    users = list(set(users))  # remove repeated users

    tweet_data = {}

    print('Getting {} tweets from each of {} {}-year old users.'.format(count, len(users), age))

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
        
        tweet_id = [tweet.id for tweet in tweets]
        tweet_permalink = [tweet.permalink for tweet in tweets]
        tweet_username = [tweet.username for tweet in tweets]
        tweet_to = [tweet.to for tweet in tweets]
        tweet_text = [tweet.text for tweet in tweets]
        tweet_date = [tweet.date for tweet in tweets]
        tweet_retweets = [tweet.retweets for tweet in tweets]
        tweet_favorites = [tweet.favorites for tweet in tweets]
        tweet_mentions = [tweet.mentions for tweet in tweets]
        tweet_hashtags = [tweet.hashtags for tweet in tweets]
        tweet_geo = [tweet.id for tweet in tweets] 

        # print(user + "has " + str(len(tweet_text)) + " tweets, adding "
        #      + str(count - len(tweet_text)) + " extra empty tweets.")
        # while len(tweet_text) < count:  # add empty elements if necessary to make every sample same length
        #    tweet_text.append('')

        if len(tweet_text) < count:  # don't use sample if user has less than 'count' tweets
            continue
        else:
            print('{}: Getting tweets from {}...'.format(userNum, user))
            userNum += 1

        for i in range(len(tweet_id)):
            tweet_data[user + str(i)] = [tweet_id[i], tweet_permalink[i], tweet_username[i], tweet_to[i],
                                         tweet_text[i], tweet_date[i], tweet_retweets[i], tweet_favorites[i],
                                         tweet_mentions[i], tweet_hashtags[i], tweet_geo[i]]

    print('Done!')
    print('{} samples created out of {} total users.'.format(len(tweet_data), len(users)))

    tweet_ds = pd.DataFrame(tweet_data).transpose().reset_index()  # original dataframe has users as a row, so transpose
    tweet_ds.insert(0, 'age', [age] * tweet_ds.shape[0], True)  # add age label
    
    tweet_ds = tweet_ds.drop('index', axis=1)
    tweet_ds.columns = ['age', 'id', 'permalink', 'username', 'to', 'text', 'date', 'retweets', 'favorites', 'mentions', 'hashtags', 'geo']

    tweet_ds.to_csv('{}yo_dataset.csv'.format(age), index=False, encoding='utf-8')

    return None



def get_age_tweets(age):
    """
    Purpose:
        Gets 2000 tweets from one age group
    Args:
        age (int): Integer age
    Returns:
        None
    """
    suffix = "th"
    if age % 10 == 1:
    	suffix = 'st'
    elif age % 10 == 2:
    	suffix = 'nd'
    elif age % 10 == 3:
    	suffix = 'rd'

    text_query = 'happy {}' + suffix +' birthday'
    print("Query: " + text_query.format(age))

    count = 5000  # set number of results to fetch

    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query.format(age)).setMaxTweets(count).setSince('2019-04-22')

    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    print('{} tweets found.'.format(len(tweets)))

    tweet_user = [tweet.username for tweet in tweets]
    tweet_text = [tweet.text for tweet in tweets]
    tweet_data = pd.DataFrame({'username': tweet_user, 'tweet': tweet_text})
    tweet_data.to_csv('{}yo.csv'.format(age), index=False, encoding='utf-8') 

    return None


def main():
    age, end_age, number = get_inputs()
    age_list = find_ages(age, end_age)

    # Make data directory if it does not exist
    data_path = os.path.dirname(os.path.abspath(__file__)) + '/data'
    os.makedirs(data_path, exist_ok=True)

    # Change working directory to data directory
    os.chdir(data_path)
    
    for age in age_list:
        get_tweetset(age, int(number), data_path)
    return None


if __name__ == '__main__':
    main()
