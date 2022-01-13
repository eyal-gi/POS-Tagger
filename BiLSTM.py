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
                 dropout,
                 pad_idx,
                 bidirectional=True):
        # Constructor
        super().__init__()

        # save model params
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.pad_idx = pad_idx

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim, padding_idx=pad_idx)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
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

    def get_params_dict(self):
        return {'input_dimension': self.input_dim,
                'embedding_dimension': self.embedding_dim,
                'num_of_layers': self.n_layers,
                'output_dimension': self.output_dim,
                'hidden_dim': self.hidden_dim,
                'dropout': self.dropout,
                'pad_idx': self.pad_idx
                }

    def categorical_accuracy(self, preds, y, tag_pad_idx):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
        non_pad_elements = (y != tag_pad_idx).nonzero()
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        return correct.sum() / y[non_pad_elements].shape[0]

    def _train(self, iterator, optimizer, criterion, tag_pad_idx):
        epoch_loss = 0
        epoch_acc = 0

        self.train()

        # for sample_id, batch in enumerate(iterator.batches):
        for x, y in iterator:
            text = batch[sample_id]['text']
            tags = batch.udtags

            optimizer.zero_grad()

            # text = [sent len, batch size]

            predictions = self(text)

            # predictions = [sent len, batch size, output dim]
            # tags = [sent len, batch size]

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            # predictions = [sent len * batch size, output dim]
            # tags = [sent len * batch size]

            loss = criterion(predictions, tags)

            acc = self.categorical_accuracy(predictions, tags, tag_pad_idx)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def _evaluate(self, iterator, criterion, tag_pad_idx):
        epoch_loss = 0
        epoch_acc = 0

        self.eval()

        with torch.no_grad():
            for batch in iterator:
                text = batch.text
                tags = batch.udtags

                predictions = self(text)

                predictions = predictions.view(-1, predictions.shape[-1])
                tags = tags.view(-1)

                loss = criterion(predictions, tags)

                acc = self.categorical_accuracy(predictions, tags, tag_pad_idx)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def _epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def fit(self, train_iterator, optimizer, criterion, TAG_PAD_IDX, n_epochs=8, valid_iterator=None, verbose=1):
        """
        Fits the model to a given train set. Can be fitted with validation set as well to see accuracy and loss
        on validation throughout the training.
        :param train_iterator: of type DataLaoder. The train data (with labels)
        :param optimizer: The optimizer to use (from pytorch functions)
        :param criterion: The models criterion
        :param TAG_PAD_IDX: The index of padding
        :param n_epochs: Number of epochs to train on
        :param valid_iterator: of type DataLaoder. Validation data (with labels)
        :param verbose: Print data during training.
        :return: history. a dictionary of the models accuracy and loss of both train and validation data
                during the data fit.
        """
        history = {'accuracy': [], 'val_accuracy': [],
                   'loss': [], 'val_loss': []
                   }

        for epoch in range(n_epochs):

            start_time = time.time()

            train_iterator.create_batches()

            train_loss, train_acc = self._train(train_iterator, optimizer, criterion, TAG_PAD_IDX)
            history['accuracy'].append(train_acc)
            history['loss'].append(train_loss)

            if valid_iterator is not None:
                valid_loss, valid_acc = self._evaluate(valid_iterator, criterion, TAG_PAD_IDX)
                history['val_accuracy'].append(valid_acc)
                history['val_loss'].append(valid_loss)

            end_time = time.time()

            epoch_mins, epoch_secs = self._epoch_time(start_time, end_time)

            if verbose == 1:
                print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
                if valid_iterator is not None:
                    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


class ConvertDataset(Dataset):
    """
    Create an instances of pytorch Dataset from lists.
    """

    def __init__(self, x, y):
        # data loading
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return {'text': self.x[index], 'tags': self.y[index]}

    def __len__(self):
        return len(self.x)
