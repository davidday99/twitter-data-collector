#!/usr/bin/env python3

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
        args.age     (list): List of ages to get
        args.end_age (str): End age if requiring age range
        args.count   (str): Number of tweets to get
    """
    # NOTE -e will add on ages starting from last age in `-a` list
    parser = argparse.ArgumentParser(description='Request tweets from GetOldTweets3 API')
    parser.add_argument('-a', '--age', dest='age', action='append', help='Age to get tweets for')
    parser.add_argument('-e', '--end', dest='end_age', default=-1, help='End age to get tweets for')
    parser.add_argument('-n', '--number', dest='number', default=10, help='Number of tweets to get')
    args = parser.parse_args()

    return args.age, args.end_age, args.number



def find_ages(ages, end_age):
    """
    Purpose:
        Produce list of ages to get from input params
    Args:
        ages    (list): List of age strings
        end_age  (str): Last age string to find data for
    Returns:
        age_list    (list): List of integer ages to find tweets for 
    """
    int_age = []
    if len(ages) is 0:
        int_age = [18]
    for age in ages:
        int_age.append(int(age))

    if len(ages) is 1 or end_age is -1:
        return int_age

    last_age = max(int_age)
    age_diff = int(end_age) - last_age
    if age_diff < 0:
        sys.exit("Last input age exceeds input age range")

    for i in range(1, age_diff+1):
        int_age.append(last_age+i)

    return int_age


def get_tweetset(age, count):
    """
    Purpose:
        Gets tweets for one age
    Args:
        age     (int): age used as a label for each data sample that is created,
                       *THIS SHOULD MATCH age IN birthdaytweetscraper.py*
        count   (int): number of tweets to scrape from each user
    Returns:
        None
    """
    FIFTEEN_MIN_IN_SEC = 15 * 65  # add a little extra time to be safe
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

    return None



def get_age_tweets():
    """
    Purpose:
        Gets 2000 tweets from one age group
    Args:
        age (int): Integer age
    Returns:
        None
    """
    count = 2000  # set number of results to fetch

    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query.format(age)).setMaxTweets(count)

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
    print(age_list)
    # Make data directory if it does not exist
    path_data = os.path.dirname(os.path.abspath(__file__)) + '/data'
    os.makedirs(path_data, exist_ok=True)
    print(path_data)

    for age in age_list:
        get_tweetset(age, int(number))
    return None


if __name__ == '__main__':
    main()