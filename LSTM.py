import torch
import torch.nn as nn
from torchtext.legacy import data


class BiLSTM(nn.Module):
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
                 input_rep,
                 bidirectional=True):
        # Constructor
        super().__init__()

        # save model params
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim if input_rep == 0 else embedding_dim + 3
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.input_rep = input_rep

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim, padding_idx=pad_idx)

        # lstm layer
        self.lstm = nn.LSTM(self.embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0)

        # dense layer
        self.fc = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, output_dim)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, features=None):
        """
        Feed forward the inputs data to the network's layers
        :param text: Input text
        :return: predictions
        """
        # ## text = [sent_length, batch size]
        embedded = self.dropout(self.embedding(text))

        if self.input_rep == 1 and features is not None:
            features_tensor = self.create_features_tensor(features) if features.ndim != 3 else features
            embedded = torch.cat([embedded, features_tensor], dim=-1)

        # ## embedded = [sent len, batch size, emb dim]

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
        """
        Returns the model parameters
        :return: dictionary
        """
        return {'input_dimension': self.input_dim,
                'embedding_dimension': self.embedding_dim,
                'num_of_layers': self.n_layers,
                'output_dimension': self.output_dim,
                'hidden_dim': self.hidden_dim,
                'dropout': self.dropout,
                'input_rep': self.input_rep,
                'pad_idx': self.pad_idx
                }

    def get_input_rep(self):
        """ Returns the model input representation (BiLSTM/CBLSTM)"""
        return self.input_rep

    def create_features_tensor(self, features):
        """Returns the features list as a tensor of vector representation"""

        # find the numerical representation of each case
        lookup = self.features_field.vocab.itos
        # create new lookup, replace case names with vector representation
        new_lookup = {0: self._feature_vec(lookup[0]),
                      1: self._feature_vec(lookup[1]),
                      2: self._feature_vec(lookup[2]),
                      3: self._feature_vec(lookup[3]),
                      4: self._feature_vec(lookup[4])
                      }

        # create the new tensor with vectors instead of numeric
        features_tensor = []
        for row in features:
            tensor_row = []
            for element in row:
                tensor_row.append(new_lookup[element.item()])
            features_tensor.append(tensor_row)

        return torch.Tensor(features_tensor)

    def set_features_field(self, field):
        """ Set the model's features field for inner use in forward method"""
        self.features_field = field

    def _feature_vec(self, name):
        """Returns a vector representation of a case base on the name (str) """

        if name == '<pad>':
            return [0, 0, 0]
        elif name == 'lower':
            return [1, 0, 0]
        elif name == 'upper':
            return [0, 1, 0]
        elif name == 'leading':
            return [0, 0, 1]
        elif name == 'other':
            return [1, 1, 1]


class SequenceTaggingDataset(data.Dataset):
    """Defines a dataset for sequence tagging. Examples in this dataset
    contain paired lists -- paired list of words and tags.

    For example, in the case of part-of-speech tagging, an example is of the
    form
    [I, love, PyTorch, .] paired with [PRON, VERB, PROPN, PUNCT]

    See torchtext/test/sequence_tagging.py on how to use this class.
    """

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, columns, fields, encoding="utf-8", separator="\t", **kwargs):
        examples = []
        # for 2 fields data sets (text, tags) [BiLSTM]
        if len(columns) == 2:
            for words, labels in zip(columns[0], columns[-1]):
                examples.append(data.Example.fromlist([words, labels], fields))
        # for 3 fields data sets (text, tags, features) [CBiLSTM]
        elif len(columns) == 3:
            for words, labels, cases in zip(columns[0], columns[1], columns[-1]):
                examples.append(data.Example.fromlist([words, labels, cases], fields))
        else:
            raise Exception("Sorry, this amount of fields is not supported")
        super(SequenceTaggingDataset, self).__init__(examples, fields,
                                                     **kwargs)
