import pandas as pd
import numpy as np

data_all = pd.read_csv('data/data_raw_clean.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)
data_all = data_all.drop('id', axis=1).drop('geo', axis=1).drop('permalink', axis=1)

df = pd.DataFrame(columns=['handle', 'tweets_text', 'followers', 'following', 'follower_following_ratio',
                           'avg_favorites', 'min_favorites', 'max_favorites', 'avg_retweets', 'min_retweets',
                           'max_retweets', 'hashtags', 'hashtags_per_tweet', 'mentions_per_tweet', 'avg_timedelta_hrs',
                           'avg_word_count', 'age_group'])

for idx, row in data_all.iterrows():
    handle = row['username']
    # Add if user is not in new dataframe
    if handle not in df['handle'].values:
        user_data = data_all.loc[data_all['username'] == handle]
        new_row = {'handle': handle,
                   'tweets_text': user_data['text'].str.cat(sep='\t'),
                   'followers': row['Followers'],
                   'following': row['Following'],
                   'follower_following_ratio': row['Follower_Following_Ratio'],
                   'avg_favorites': user_data['favorites'].mean(),
                   'min_favorites': user_data['favorites'].min(),
                   'max_favorites': user_data['favorites'].max(),
                   'avg_retweets': user_data['retweets'].mean(),
                   'min_retweets': user_data['retweets'].min(),
                   'max_retweets': user_data['retweets'].max(),
                   'hashtags': user_data['hashtags'].str.cat(sep=' '),
                   'hashtags_per_tweet': len(user_data['hashtags'].str.cat(sep=' ').split()) / user_data.shape[0],
                   'mentions_per_tweet': len((user_data.loc[user_data['mentions'].notna()])['mentions']) / user_data.shape[0],
                   'avg_word_count': len(user_data['text'].str.cat(sep=' ').split())
                   }

        # Calculate tweet frequency
        dt_data = pd.to_datetime(user_data['date']).sort_values()
        dt_deltas = dt_data - dt_data.shift()
        avg_delta = dt_deltas.mean() / np.timedelta64(1, 'h')
        new_row['avg_timedelta_hrs'] = avg_delta

        # Sort into age group
        age = row['age']
        if age < 25:
            age_group = 0
        elif 26 <= age <= 40:
            age_group = 1
        elif 41 <= age <= 55:
            age_group = 2
        elif 56 <= age <= 76:
            age_group = 3
        else:
            continue    # ignore >=77yo
        new_row['age_group'] = age_group
        print(new_row)
        df = df.append(new_row, ignore_index=True)

print(df.shape)
print(df.head())
df.to_csv('data/profile_data_all.csv')
