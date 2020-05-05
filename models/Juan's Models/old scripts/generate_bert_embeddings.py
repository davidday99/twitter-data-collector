import random
import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, BertPreTrainedModel, BertConfig
from keras.preprocessing.sequence import pad_sequences
import torch

train_df = pd.read_csv('data/train_w_logits.csv')
test_df = pd.read_csv('data/test_w_logits.csv')
data_all = pd.concat([train_df, test_df])

# Load pretrained model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
model.cuda()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = []
for tweet in data_all.tweets_text.values:
    encoded = tokenizer.encode(tweet, add_special_tokens=True, max_length=512)
    input_ids.append(encoded)
input_ids = pad_sequences(input_ids, maxlen=512, dtype='long', value=0, padding='post', truncating='post')

attention_masks = []
for tweet in input_ids:
    mask = [int(token_id > 0) for token_id in tweet]
    attention_masks.append(mask)

input_ids = np.array(input_ids)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
output_encodings = []
model.eval()

for i in range(len(input_ids)):
    encoded = input_ids[i]
    attention_masks = [int(token_id > 0) for token_id in encoded]
    encoded = torch.tensor(encoded).to(device)
    encoded = encoded.reshape(1, 512)
    segment_tensor = torch.tensor([[1] * 512]).to(device)
    attention_masks = torch.tensor(attention_masks).to(device)
    attention_masks = attention_masks.reshape(1, 512)

    with torch.no_grad():
        last_hidden_states = model(encoded, token_type_ids=segment_tensor, attention_mask=attention_masks)
    features = last_hidden_states[0][:, 0, :].cpu().numpy()
    output_encodings.append(features)

output_encodings = np.array(output_encodings)
output_encodings = output_encodings.reshape(data_all.shape[0], 768)

for i in range(768):
    feat_name = 'embed' + str(i)
    data_all[feat_name] = output_encodings[:, i]

train_df = data_all[:train_df.shape[0]]
test_df = data_all[train_df.shape[0]:]

print(train_df.head())
print(test_df.head())

test_df.to_csv('data/test_w_logits_and_embeddings.csv', index=False)
train_df.to_csv('data/train_w_logits_and_embeddings.csv', index=False)
