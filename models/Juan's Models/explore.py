import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

data_all = pd.read_csv('../data/data_raw_clean.csv')
print(data_all.shape)
print(data_all[data_all['Followers'] < 5000].shape)

avg_likes = []
for age in range(18, 91):
    avg_likes.append(data_all[data_all.age == age]['Follower_Following_Ratio'].mean())

plt.bar(range(18, 91), avg_likes)
plt.show()

plt.bar(range(18, 91), data_all['age'].value_counts())
plt.show()

text = ''
for t in data_all['text']:
    text += t + ' '

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()