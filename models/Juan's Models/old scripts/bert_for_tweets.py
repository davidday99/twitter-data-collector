import random
import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, BertPreTrainedModel, BertConfig
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, Subset, DataLoader, RandomSampler, SequentialSampler, TensorDataset


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train_one_epoch(model, train_dataloader, optimizer, scheduler):
    print("*************STARTING EPOCH*************")
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    total_loss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Epoch
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 100 == 0:
            print("Batch " + str(step) + ' of ' + str(len(train_dataloader)))
        b_input_ids = batch[0].to(device)
        b_input_masks = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_masks, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print("Done training epoch. Average training loss: " + str(avg_train_loss))
    return avg_train_loss


def eval_tweets(model, test_dataloader):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictions, true_labels = [], []
    eval_accuracy, nb_eval_steps = 0, 0
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        if (step % 100 == 0):
            print("batch " + str(step) + ' of ' + str(len(test_dataloader)))
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.append(logits)
        true_labels.append(label_ids)
        temp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += temp_eval_accuracy
        nb_eval_steps += 1
    print("Individual tweet accuracy: " + str(eval_accuracy / nb_eval_steps))
    return predictions, true_labels

def eval_profile(predictions, tweet_eval_set):
    logits = [item for sublist in predictions for item in sublist]
    logits = np.array(logits)
    tweet_eval_set = tweet_eval_set.assign(Logit0=logits[:, 0])
    tweet_eval_set = tweet_eval_set.assign(Logit1=logits[:, 1])
    tweet_eval_set = tweet_eval_set.assign(Logit2=logits[:, 2])
    tweet_eval_set = tweet_eval_set.assign(Logit3=logits[:, 3])

    user_labels = [
        np.sum(tweet_eval_set.loc[tweet_eval_set['username'] == user][['Logit0', 'Logit1', 'Logit2', 'Logit3']]) for
        user in df_test.handle.values]
    user_labels = np.argmax(user_labels, axis=1).flatten()
    true_labels = df_test.age_group.values

    accuracy = np.sum(user_labels == true_labels) / len(true_labels)
    print("Test accuracy by user (average of logits): " + str(accuracy))
    return accuracy

print("Starting training script. Reading input data")
# ********************************** Read CSVs **********************************
df_train = pd.read_csv('data/profile_data_train.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)
df_test = pd.read_csv('data/profile_data_test.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)
df_tweets_all = pd.read_csv('data/data_raw_clean.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)


print("Preparing input and test data")
# ********************************** Train data prep **********************************
train_handles = df_train.handle.values
train_tweet_set = df_tweets_all.loc[df_tweets_all['username'].isin(train_handles)]
train_tweet_set = train_tweet_set[['username', 'text']]
train_tweets = train_tweet_set.text.values
train_labels = [df_train.loc[df_train['handle'] == name].age_group.values[0] for name in
                train_tweet_set.username.values]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = []

for tweet in train_tweets:
    encoded = tokenizer.encode(tweet, add_special_tokens=True, max_length=512)
    input_ids.append(encoded)

input_ids = pad_sequences(input_ids, maxlen=512, dtype='long', value=0, padding='post', truncating='post')

attention_masks = []
for tweet in input_ids:
    mask = [int(token_id > 0) for token_id in tweet]
    attention_masks.append(mask)

batch_size = 8
train_inputs = torch.tensor(input_ids)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(attention_masks)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# ********************************** Test data prep **********************************
test_handles = df_test.handle.values
test_tweet_set = df_tweets_all.loc[df_tweets_all['username'].isin(test_handles)]
test_tweet_set = test_tweet_set[['username', 'text']]
test_tweets = test_tweet_set.text.values
test_labels = [df_test.loc[df_test['handle'] == name].age_group.values[0] for name in test_tweet_set.username.values]

test_input_ids = []

for tweet in test_tweets:
    encoded = tokenizer.encode(tweet, add_special_tokens=True, max_length=512)
    test_input_ids.append(encoded)

test_input_ids = pad_sequences(test_input_ids, maxlen=512, dtype='long', value=0, padding='post', truncating='post')
test_attention_masks = []
for tweet in test_input_ids:
    mask = [int(token_id > 0) for token_id in tweet]
    test_attention_masks.append(mask)

test_inputs = torch.tensor(test_input_ids)
test_labels = torch.tensor(test_labels)
test_masks = torch.tensor(test_attention_masks)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# ********************************** Model training and eval **********************************
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4, output_attentions=False,
                                                      output_hidden_states=False)
model.cuda()

epochs = 10
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(epochs):
    print("Epoch " + str(epoch + 1))
    loss = train_one_epoch(model, train_dataloader, optimizer, scheduler)
    print("Eval on test tweets")
    raw_predictions, raw_labels = eval_tweets(model, test_dataloader)
    print("Eval on test profiles")
    accuracy = eval_profile(raw_predictions, test_tweet_set)

# Save model

output_dir = './bert_tweet_classifier/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
