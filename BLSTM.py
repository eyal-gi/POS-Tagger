import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.data import Field, LabelField, BucketIterator, TabularDataset
from torchtext.legacy import datasets

import spacy
import numpy as np

import time
import random

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, \
    classification_report
import string
import pickle
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
# from tabulate import tabulate

import tagger

tagger.use_seed()
TEXT = Field(lower=True)
UD_TAGS = Field(unk_token=None)
# PTB_TAGS = Field(unk_token=None)

fields = (("text", TEXT), ("udtags", UD_TAGS), (None, None))
train_data, valid_data, _ = datasets.UDPOS.splits(fields)

MIN_FREQ = 2
TEXT.build_vocab(train_data,
                 min_freq=MIN_FREQ,
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_)

UD_TAGS.build_vocab(train_data)


BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator = BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE,
    device=device)


class LSTM(nn.Module):
    """
    Creates an instance of LSTM model.
    Inner methods: model fit, model evaluation.
    """

    # define all the layers used in model
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout,
                 pad_idx):
        # Constructor
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim, padding_idx=pad_idx)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0)

        # dense layer
        self.fc = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, output_dim)

        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # activation function
        self.act = nn.Sigmoid()

    def forward(self, text):
        """
        Feed forward the inputs data to the network's layers
        :param text: Input text
        :return: predictions
        """
        # ## text = [sent_length, batch size]
        embedded = self.dropout(self.embedding(text))

        # ## embedded = [sent len, batch size, emb dim]

        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)

        # ## pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        # ## outputs holds the backward and forward hidden states in the final layer
        # ## hidden and cell are the backward and forward hidden and cell states at the final time-step
        # ## output = [sent len, batch size, hid dim * n directions]
        # ## hidden, cell = [batch size, num layers * num directions,hid dim]

        # we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))
        # predictions = [sent len, batch size, output dim]

        return predictions


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = len(UD_TAGS.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = LSTM(input_dim=INPUT_DIM,
             embedding_dim=EMBEDDING_DIM,
             hidden_dim=HIDDEN_DIM,
             output_dim=OUTPUT_DIM,
             n_layers=N_LAYERS,
             bidirectional=BIDIRECTIONAL,
             dropout=DROPOUT,
             pad_idx=PAD_IDX)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)


model.apply(init_weights)

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
# set pad tag embedding to 0
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters())
TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
print(TAG_PAD_IDX)

criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)

model = model.to(device)
criterion = criterion.to(device)


def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / y[non_pad_elements].shape[0]


def train(model, iterator, optimizer, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        text = batch.text
        tags = batch.udtags

        optimizer.zero_grad()

        # text = [sent len, batch size]

        predictions = model(text)

        # predictions = [sent len, batch size, output dim]
        # tags = [sent len, batch size]

        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)

        # predictions = [sent len * batch size, output dim]
        # tags = [sent len * batch size]

        loss = criterion(predictions, tags)

        acc = categorical_accuracy(predictions, tags, tag_pad_idx)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            tags = batch.udtags

            predictions = model(text)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            loss = criterion(predictions, tags)

            acc = categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# N_EPOCHS = 17
N_EPOCHS = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        print(f'best epochs num is {epoch+1}')
        # torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
