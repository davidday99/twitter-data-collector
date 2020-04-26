from bs4 import BeautifulSoup
import requests
import pandas as pd

data_all = pd.read_csv('data/25yo_dataset.csv')
# data_all = pd.read_csv('data/18yo_dataset.csv')

for age in range(19, 94):
    try:
        csv_name = 'data/' + str(age) + 'yo_dataset.csv'
        temp = pd.read_csv(csv_name)
        data_all = pd.concat([data_all, temp])
        print(csv_name)
    except Exception as e:
        print(e)
        continue

print(data_all.shape)
print(data_all.head())

list_followers = []
list_following = []
list_ratio = []

for idx, row in data_all.iterrows():
    handle = row['username']
    print('scraping for ' + str(handle))
    temp = requests.get('https://twitter.com/' + str(handle))
    bs = BeautifulSoup(temp.text,'lxml')
    try:
        follow_box = bs.find('li',{'class':'ProfileNav-item ProfileNav-item--followers'})
        followers = follow_box.find('a').find('span',{'class':'ProfileNav-value'})
        print("Number of followers: {} ".format(followers.get('data-count')))
        list_followers.append(int(followers.get('data-count')))
        following_box = bs.find('li',{'class':'ProfileNav-item ProfileNav-item--following'})
        following = following_box.find('a').find('span',{'class':'ProfileNav-value'})
        print("Number following: {} ".format(following.get('data-count')))
        list_following.append(int(following.get('data-count')))
        list_ratio.append((float(int(followers.get('data-count'))/int(following.get('data-count')))))
        print('ratio = ' + str(int(followers.get('data-count'))/int(following.get('data-count'))))
    except Exception as e:
        print(e)
        print('Account name not found...')
    print()

data_all = data_all.assign(Followers=list_followers)
data_all = data_all.assign(Following=list_following)
data_all = data_all.assign(Follower_Following_Ratio=list_ratio)


print(data_all.shape)
print(data_all.head())

data_all.to_csv('data/data_all_raw.csv')
