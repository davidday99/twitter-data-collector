import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

data_all = pd.read_csv('data/data_raw_clean.csv')
profile_df = pd.read_csv('data/profile_data_all.csv')

lengths = data_all.text.values
lengths = [len(x) for x in lengths]
plt.hist(lengths)
plt.show()
print(np.max(lengths))
print(np.sum([int(lengths[i] >= 256) for i in range(len(lengths))]) / len(lengths))

text = ''
for t in data_all['text']:
    text += t + ' '

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()