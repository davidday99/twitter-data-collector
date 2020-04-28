from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np


print("\n************************Combining all datasets************************\n")

data_all = pd.read_csv('data/18yo_dataset.csv')

for age in range(19, 91):
    try:
        csv_name = 'data/' + str(age) + 'yo_dataset.csv'
        temp = pd.read_csv(csv_name)
        data_all = pd.concat([data_all, temp])
        print(csv_name)
    except Exception as e:
        print(e)
        continue

data_all = data_all.assign(Followers=np.zeros(data_all.shape[0]))
data_all = data_all.assign(Following=np.zeros(data_all.shape[0]))
data_all = data_all.assign(Follower_Following_Ratio=np.zeros(data_all.shape[0]))
data_all = data_all.reset_index()

print("New DataAll Shape")
print(data_all.shape)
print(data_all.head())
data_all.to_csv('data/data_all_raw.csv')

print("\n************************Populating metadata************************\n")

for idx, row in data_all.iterrows():
    if data_all.loc[data_all.index[idx], 'Followers'] == 0:
        handle = row['username']
        print('scraping for ' + str(handle))
        temp = requests.get('https://twitter.com/' + str(handle))
        bs = BeautifulSoup(temp.text,'lxml')
        print()
        try:
            print("Scraping data for " + handle)
            follow_box = bs.find('li',{'class':'ProfileNav-item ProfileNav-item--followers'})
            followers = follow_box.find('a').find('span',{'class':'ProfileNav-value'})
            print("Number of followers: {} ".format(followers.get('data-count')))
            data_all.loc[data_all['username'] == handle, 'Followers'] = int(followers.get('data-count'))
            following_box = bs.find('li',{'class':'ProfileNav-item ProfileNav-item--following'})
            following = following_box.find('a').find('span',{'class':'ProfileNav-value'})
            print("Number following: {} ".format(following.get('data-count')))
            data_all.loc[data_all['username'] == handle, 'Following'] = int(following.get('data-count'))
            data_all.loc[data_all['username'] == handle, 'Follower_Following_Ratio'] = int(followers.get('data-count'))/int(following.get('data-count'))
            print('ratio = ' + str(int(followers.get('data-count'))/int(following.get('data-count'))))
        except Exception as e:
            print(e)
            print('Unable to scrape data for ' + handle)

print("************************Success, saving csv************************")

data_all.to_csv('data/data_all_raw.csv')
