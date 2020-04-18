#!/usr/bin/env python3

# First party packages
import argparse
import re
import sys
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
    parser.add_argument('-e', '--end', dest='end_age', help='End age to get tweets for')
    parser.add_argument('-n', '--number', dest='number', help='Number of tweets to get')
    args = parser.parse_args()

    return args.age, args.end_age, args.number



def find_ages(ages, end_age):
    """
    Purpose:
        Produce list of ages to get from input params
    Args:
        ages    (list): List of age integers
        end_age  (int): Last age to find data for
    Returns:
        age_list    (list): List of ages to find tweets for 
    """
    last_age = ages[-1]
    age_diff = end_age - last_age
    if (age_diff < 0):
        sys.exit("Last input age exceeds input age range")

    for i in range(1, age_diff+1):
        ages.append(last_age+1)

    return ages



def get_tweets(age, count):
    """
    Purpose:
        Gets tweets for one age
    Args:
        age     (int): Age of tweets to get
        count   (int): Number of tweets to get
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

    return None


def main():
    # age used as a label for each data sample that is created, *THIS SHOULD MATCH age IN birthdaytweetscraper.py*
    age = [18]
    end_age = -1
    # number of tweets to scrape from each user
    count = 10

    age, end_age, number = get_inputs()
    age_list = find_ages(age, end_age)
    for age in age_list:
        get_tweets(age, number)
    return None


if __name__ == '__main__':
    main()