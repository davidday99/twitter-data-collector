import random
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
import torch
from torch import nn, optim
from torch.utils.data import Subset, DataLoader, RandomSampler, SequentialSampler, TensorDataset


def accuracy(logits, labels):
    preds = np.argmax(logits, axis=1).flatten()
    labels = labels.flatten()
    return np.sum(preds == labels) / len(labels)


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
        batch_input_ids = batch[0].to(device)
        batch_input_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)
        model.zero_grad()
        outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_masks, labels=batch_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print("Done training epoch. Average training loss: " + str(avg_train_loss))
    return avg_train_loss


def eval_profiles(model, test_dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions, true_labels = [], []

    eval_accuracy, eval_steps = 0, 0

    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        if step % 100 == 0:
            print("batch " + str(step) + ' of ' + str(len(test_dataloader)))
        batch_input_ids, batch_input_mask, batch_labels = batch
        with torch.no_grad():
            outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        labels = batch_labels.to('cpu').numpy()
        predictions.append(logits)
        true_labels.append(labels)
        eval_accuracy += accuracy(logits, labels)
        eval_steps += 1

    print("Calculated accuracy on eval set: " + str(eval_accuracy / eval_steps))
    return predictions, true_labels


print("Starting training script. Reading input data")

# ********************************** Read Input CSVs **********************************
df_train = pd.read_csv('data/profile_data_train.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)
df_test = pd.read_csv('data/profile_data_test.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)

print("Starting data prep")
# ********************************** Train data prep **********************************

# Assign training logits
df_train = df_train.assign(Logit0=np.zeros(df_train.shape[0]))
df_train = df_train.assign(Logit1=np.zeros(df_train.shape[0]))
df_train = df_train.assign(Logit2=np.zeros(df_train.shape[0]))
df_train = df_train.assign(Logit3=np.zeros(df_train.shape[0]))

# Split train dataframe into 8 stratified chunks
X_train = df_train.drop('age_group', axis=1)
Y_train = df_train.age_group.values.tolist()
X_test = df_test.drop('age_group', axis=1)
Y_test = df_test.age_group.values.tolist()

# First split: 50-50
X_1, X_2, Y_1, Y_2 = train_test_split(X_train, Y_train, stratify=Y_train, test_size=0.5, random_state=42)

# Second split: 25-25-25-25
X_1, X_3, Y_1, Y_3 = train_test_split(X_1, Y_1, stratify=Y_1, test_size=0.5, random_state=42)
X_2, X_4, Y_2, Y_4 = train_test_split(X_2, Y_2, stratify=Y_2, test_size=0.5, random_state=42)

# Third split: 12.5 x 8
X_1, X_5, Y_1, Y_5 = train_test_split(X_1, Y_1, stratify=Y_1, test_size=0.5, random_state=42)
X_2, X_6, Y_2, Y_6 = train_test_split(X_2, Y_2, stratify=Y_2, test_size=0.5, random_state=42)
X_3, X_7, Y_3, Y_7 = train_test_split(X_3, Y_3, stratify=Y_3, test_size=0.5, random_state=42)
X_4, X_8, Y_4, Y_8 = train_test_split(X_4, Y_4, stratify=Y_4, test_size=0.5, random_state=42)

X_1 = X_1.reset_index(drop=True)
X_2 = X_2.reset_index(drop=True)
X_3 = X_3.reset_index(drop=True)
X_4 = X_4.reset_index(drop=True)
X_5 = X_5.reset_index(drop=True)
X_6 = X_6.reset_index(drop=True)
X_7 = X_7.reset_index(drop=True)
X_8 = X_8.reset_index(drop=True)

X_splits = [X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8]
Y_splits = [Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ********************************** Generate logits on holdouts **********************************

# Generate logits for each holdout by training on the others
epochs = 10
batch_size = 4

for holdout_idx in range(8):
    print("********************************************************************************")
    print("********************* GENERATING LOGITS FOR HOLDOUT " + str(holdout_idx + 1) + " OF 8 *********************")
    print("********************************************************************************\n")
    X_temp, Y_temp = [], []
    X_holdout, Y_holdout = [], []
    # Generate temporary X and Y for training and holdout
    for idx in range(8):
        if idx != holdout_idx:
            X_temp = X_temp + X_splits[idx].tweets_text.values.tolist()
            Y_temp = Y_temp + Y_splits[idx]
        else:
            X_holdout = X_splits[idx].tweets_text.values.tolist()
            Y_holdout = Y_splits[idx]

    # Encode and pad tokens
    input_ids, holdout_input_ids = [], []

    for tweet in X_temp:
        encoded = tokenizer.encode(tweet, add_special_tokens=True, max_length=512)
        input_ids.append(encoded)
    input_ids = pad_sequences(input_ids, maxlen=512, dtype='long', value=0, padding='post', truncating='post')

    for tweet in X_holdout:
        encoded = tokenizer.encode(tweet, add_special_tokens=True, max_length=512)
        holdout_input_ids.append(encoded)
    holdout_input_ids = pad_sequences(holdout_input_ids, maxlen=512, dtype='long', value=0, padding='post',
                                      truncating='post')

    # Attention masks to ignore padded tokens
    attention_masks, holdout_attention_masks = [], []
    for tweet in input_ids:
        mask = [int(token_id > 0) for token_id in tweet]
        attention_masks.append(mask)

    for tweet in holdout_input_ids:
        mask = [int(token_id > 0) for token_id in tweet]
        holdout_attention_masks.append(mask)

    # Prep torch data
    train_inputs = torch.tensor(input_ids)
    train_labels = torch.tensor(Y_temp)
    train_masks = torch.tensor(attention_masks)
    holdout_inputs = torch.tensor(holdout_input_ids)
    holdout_labels = torch.tensor(Y_holdout)
    holdout_masks = torch.tensor(holdout_attention_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    holdout_data = TensorDataset(holdout_inputs, holdout_masks, holdout_labels)
    holdout_dataloader = DataLoader(holdout_data, sampler=None, batch_size=batch_size)

    # Load empty model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4, output_attentions=False,
                                                          output_hidden_states=False)
    model.cuda()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train over epochs
    losses = []
    for epoch in range(epochs):
        print("Epoch " + str(epoch + 1))
        loss = train_one_epoch(model, train_dataloader, optimizer, scheduler)
        losses.append(loss)

    predictions, true_labels = eval_profiles(model, holdout_dataloader)

    # Assign logits to dataframe
    logits = [item for sublist in predictions for item in sublist]
    logits = np.array(logits)

    # Save logits to df
    X_splits[holdout_idx] = X_splits[holdout_idx].reset_index(drop=True)
    for idx, row in X_splits[holdout_idx].iterrows():
        handle = row['handle']
        df_train.loc[df_train['handle'] == handle, 'Logit0'] = logits[idx][0]
        df_train.loc[df_train['handle'] == handle, 'Logit1'] = logits[idx][1]
        df_train.loc[df_train['handle'] == handle, 'Logit2'] = logits[idx][2]
        df_train.loc[df_train['handle'] == handle, 'Logit3'] = logits[idx][3]

    print(df_train.head())  # Logits should be partially populated

    del model  # Free cuda memory, prevent information leakage

# Save logits
df_train.to_csv('data/train.csv', index=False)

print("\n\nDone producing logits for holdouts. Generating logits for test set\n\n")

# ********************************** Generate logits on test set **********************************

input_ids = []

for tweet in X_train.tweets_text.values.tolist():
    encoded = tokenizer.encode(tweet, add_special_tokens=True, max_length=512)
    input_ids.append(encoded)

input_ids = pad_sequences(input_ids, maxlen=512, dtype='long', value=0, padding='post', truncating='post')
attention_masks = []
for tweet in input_ids:
    mask = [int(token_id > 0) for token_id in tweet]
    attention_masks.append(mask)

train_inputs = torch.tensor(input_ids)
train_labels = torch.tensor(Y_train)
train_masks = torch.tensor(attention_masks)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Load empty model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4, output_attentions=False,
                                                      output_hidden_states=False)
model.cuda()
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Train over epochs
for epoch in range(epochs):
    print("Epoch " + str(epoch + 1))
    loss = train_one_epoch(model, train_dataloader, optimizer, scheduler)

# Generate test inputs
test_input_ids = []

for tweet in X_test.tweets_text.values.tolist():
    encoded = tokenizer.encode(tweet, add_special_tokens=True, max_length=512)
    test_input_ids.append(encoded)

test_input_ids = pad_sequences(test_input_ids, maxlen=512, dtype='long', value=0, padding='post', truncating='post')
test_attention_masks = []
for tweet in test_input_ids:
    mask = [int(token_id > 0) for token_id in tweet]
    test_attention_masks.append(mask)

test_inputs = torch.tensor(test_input_ids)
test_labels = torch.tensor(Y_test)
test_masks = torch.tensor(test_attention_masks)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Make predictions for test set and get logits
predictions, true_labels = eval_profiles(model, test_dataloader)

logits = [item for sublist in predictions for item in sublist]
logits = np.array(logits)
df_test = df_test.assign(Logit0=logits[:, 0])
df_test = df_test.assign(Logit1=logits[:, 1])
df_test = df_test.assign(Logit2=logits[:, 2])
df_test = df_test.assign(Logit3=logits[:, 3])

# Save logits
df_test.to_csv('data/test.csv', index=False)

# Save model
output_dir = './bert_logit_generator/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
