"""
nlp, assignment 4, 2021

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""
import copy
import math
import torch
import torch.nn as nn
import torchtext.legacy.data
import torch.optim as optim
from math import log, isfinite
from collections import Counter
import time
import random
import numpy as np
import LSTM
from torchtext.legacy.data import Field, BucketIterator

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.
def use_seed(seed=2512021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.set_deterministic(True)
    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True


# utility functions to read the corpus
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    # TODO edit the dictionary to have your own details
    return {'name': 'Eyal Ginosar', 'id': '', 'email': ''}


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append((word, tag))
        line = f.readline()
    return sentence


def load_annotated_corpus(filename):
    """ Returns a list of lists where each list represents a sentence
        where every item in the list is a tuple of word and POS tag
    """
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


# ===========================================
# ------------------------------------------
# ===========================================

START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {}  # transitions probabilities
B = {}  # emissions probabilities


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
     and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transisionCounts and emmisionCounts
    should be computed with pseudo tags and should be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transisionCounts and  emmisionCounts

    Args:
      tagged_sentences: a list of tagged sentences, each tagged sentence is a
       list of pairs (w,t), as retunred by load_annotated_corpus().

   Return:
      [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
  """
    # todo consider not doing this
    global VOCAB
    global TAGS

    train_tagged_words = [tup for sent in tagged_sentences for tup in sent]
    tags = {tag for word, tag in train_tagged_words}
    vocab = {word for word, tag in train_tagged_words}
    vocab.add(UNK)
    VOCAB = vocab
    TAGS = tags
    allTagCounts = Counter(tag for word, tag in train_tagged_words)
    perWordTagCounts = dict(Counter(tup for tup in train_tagged_words))

    tagged_sentences_dummy = copy.deepcopy(tagged_sentences)
    for sent in tagged_sentences_dummy:
        sent.insert(0, ('<s>', START))
        sent.append(('<e>', END))

    train_tagged_words_dummy = [tup for sent in tagged_sentences_dummy for tup in sent]
    tags_dummy = [tag for word, tag in train_tagged_words_dummy]

    transitionCounts = dict(Counter((tags_dummy[i], tags_dummy[i + 1]) for i in range(len(tags_dummy) - 1) if
                                    tags_dummy[i] != END and tags_dummy[i + 1] != START))
    emissionCounts = dict(Counter(tup for tup in train_tagged_words_dummy))
    allTagsCounts_dummy = dict(Counter(tag for word, tag in train_tagged_words_dummy))

    # smooth transitionCounts
    tags_list = list(allTagsCounts_dummy.keys())
    for i in range(len(tags_list)):
        for j in range(len(tags_list)):
            tup = (tags_list[i], tags_list[j])
            if tup not in transitionCounts.keys():
                transitionCounts[tup] = 1

    # smooth emissionCounts
    # for w in vocab:
    for t in tags:
        tup = (UNK, t)
        if tup not in emissionCounts.keys():
            emissionCounts[tup] = 1
        else:
            emissionCounts[tup] += 1

    # create A dict
    for key, value in transitionCounts.items():
        # print(f'P({key[1]}|{key[0]}) = {value} / {allTagsCounts_dummy[key[0]]}')
        A[key] = math.log(value / allTagsCounts_dummy[key[0]])

    # create B dict
    for key, value in emissionCounts.items():
        B[key] = math.log(value / allTagsCounts_dummy[key[1]])

    return [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        list: list of pairs
    """
    tagged_sentence = []
    for word in sentence:
        if word in VOCAB:
            # dictionary of only the (word,tag) and values
            word_tag_dict = {tag: perWordTagCounts[(word, tag)] for tag in allTagCounts if
                             (word, tag) in perWordTagCounts}

            # choose the tag with max count
            tag = max(word_tag_dict, key=word_tag_dict.get)
            tagged_sentence.append((word, tag))

        # OOV word
        else:
            # tag is sampled by a random choice from the pos tag distribution
            oov_tag = random.choices(population=list(allTagCounts.keys()), weights=list(allTagCounts.values()), k=1)[0]
            tagged_sentence.append((word, oov_tag))

    return tagged_sentence


# ===========================================
#       POS tagging with HMM
# ===========================================


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        list: list of pairs
    """

    tagged_sentence = []
    # find best sequence with viterbi algorithm
    last_item = viterbi(sentence, A, B)
    # recursively retrace the pos tagging
    pos_list = retrace(last_item)

    for word, tag in zip(sentence, pos_list):
        tagged_sentence.append((word, tag))

    return tagged_sentence


def viterbi(sentence, A, B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tupple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probability of the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): The HMM Emission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtracking.

        """
    # Hint 1: For efficiency reasons - for words seen in training there is no
    #      need to consider all tags in the tagset, but only tags seen with that
    #      word. For OOV you have to consider all tags.
    # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
    #         current list = [ the dummy item ]
    # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END

    """
    create an array Q[][] with dimension of num_states(tags)*num_words
    first column = P(state|<start>)*P(word|state)
    loop through all the words and all the states
    """
    # dynamic programing history matrix
    q_matrix = []

    # for every word in the sentence
    for w, i in zip(sentence, range(len(sentence))):
        # init empty column
        col = []
        # if a word is OOV -> assign with the dummy UNK and check all tags
        if w not in VOCAB:
            w.lower()  # look for this word when it is lower cased
            if w not in VOCAB:
                word = UNK
                tags_list = TAGS
        else:
            word = w
            # create a list of tags of only tags that appeared for this word in the train set
            tags_list = [tag for tag in TAGS if (word, tag) in B]

        # calc prob for every tag for the word
        for tag in tags_list:
            # initialization step (first column)
            if i == 0:
                r = START
                col.append((tag, r, A[(START, tag)] + B[(word, tag)]))

            # main algorithm
            else:
                tag_, r, log_prob = predict_next_best(word, tag, q_matrix[i - 1])
                col.append((tag_, r, log_prob))

        q_matrix.append(col)

    v_last = predict_next_best('<e>', END, q_matrix[-1])
    return v_last


def _print_matrix(matrix):
    for col, i in zip(matrix, range(len(matrix))):
        print(f'--------- COL {i} ----------')
        print(col)


# a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    retrace_list = []
    retrace_(end_item, retrace_list)

    # remove START and END dummies
    retrace_list = retrace_list[1:-1]
    return retrace_list[::-1]


def retrace_(item, retrace_list):
    """
    Recursive function to create the retrace list
    :param item: current item
    :param retrace_list: the retrace list
    """
    if item[1] == START:
        retrace_list.append(item[0])
        return retrace_list.append(item[1])

    retrace_list.append(item[0])
    retrace_(item[1], retrace_list)


# a suggestion for a helper function. Not an API requirement
def predict_next_best(word, tag, predecessor_list):
    """Returns a new item (tupple)
    """
    candidates = []  # candidate list for every cell (transition for every other tag))
    for prev in predecessor_list:
        # ##__prev[0]=tag, prev[1]=backtrace, prev[2]=log-prob__##
        prev_tag, _, prev_log_prob = prev
        candidates.append(prev_log_prob + A[(prev_tag, tag)] + B[(word, tag)])
    best_score = max(candidates)
    best_index = candidates.index(best_score)
    r = predecessor_list[best_index]

    return tag, r, best_score


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): the HMM emmission probabilities.
     """
    p = 0  # joint log prob. of words and tags
    prev = START
    for word, tag in sentence:
        if word not in VOCAB:
            word = UNK
        p += A[(prev, tag)]
        p += B[(word, tag)]
        prev = tag

    assert isfinite(p) and p < 0  # Should be negative. Think why!
    return p


# ===========================================
#       POS tagging with BiLSTM
# ===========================================

""" You are required to support two types of bi-LSTM:
    1. a vanilla biLSTM in which the input layer is based on simple word embeddings
    2. a case-based BiLSTM in which input vectors combine a 3-dim binary vector
        encoding case information, see
        https://arxiv.org/pdf/1510.06168.pdf
"""


# Suggestions and tips, not part of the required API
#
#  1. You can use PyTorch torch.nn module to define your LSTM, see:
#     https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
#  2. You can have the BLSTM tagger model(s) implemented in a dedicated class
#     (this could be a subclass of torch.nn.Module)
#  3. Think about padding.
#  4. Consider using dropout layers
#  5. Think about the way you implement the input representation
#  6. Consider using different unit types (LSTM, GRU,LeRU)

def initialize_rnn_model(params_d):
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
       the lstm model. The LSTM is initialized based on the specified parameters.
       thr returned dict is may have other or additional fields.

    Args:
        params_d (dict): a dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'max_vocab_size': max vocabulary size (int),
                        'min_frequency': the occurence threshold to consider (int),
                        'input_rep': 0 for the vanilla and 1 for the case-base (int),
                        'embedding_dimension': embedding vectors size (int),
                        'num_of_layers': number of layers (int),
                        'output_dimension': number of tags in tagset (int),
                        'pretrained_embeddings_fn': str,
                        'data_fn': str
                        }
                        max_vocab_size sets a constraints on the vocab dimention.
                            If the its value is smaller than the number of unique
                            tokens in data_fn, the words to consider are the most
                            frequent words. If max_vocab_size = -1, all words
                            occuring more that min_frequency are considered.
                        min_frequency privides a threshold under which words are
                            not considered at all. (If min_frequency=1 all words
                            up to max_vocab_size are considered;
                            If min_frequency=3, we only consider words that appear
                            at least three times.)
                        input_rep (int): sets the input representation. Values:
                            0 (vanilla), 1 (case-base);
                            <other int>: other models, if you are playful
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        a dictionary with the at least the following key-value pairs:
                                       {'lstm': torch.nn.Module object,
                                       input_rep: [0|1]}
        #Hint: you may consider adding the embeddings and the vocabulary
        #to the returned dict
    """
    use_seed()
    input_rep = params_d['input_rep']

    if params_d['output_dimension'] == 17:
        print('I assumed the input to be 18 (17 tags+ 1unk), please add +1 dim for unk :)')

    # ## model variables
    DROPOUT = 0.25
    HIDDEN_DIM = 128

    # ## load and prepare train data
    train_set = load_annotated_corpus(params_d['data_fn'])
    x_train, y_train = _prepare_data(train_set)
    TEXT = Field()
    UD_TAGS = Field(unk_token=None)

    # ## build words and tags vocabularies
    TEXT.build_vocab(x_train,
                     min_freq=params_d['min_frequency'],
                     # unk_init=torch.Tensor.normal_,
                     # vectors=params_d['pretrained_embeddings_fn'],
                     max_size=None if params_d['max_vocab_size'] == -1 else params_d['max_vocab_size'])

    UD_TAGS.build_vocab(y_train)

    # ## more model variables
    INPUT_DIM = len(TEXT.vocab)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    # ## initiate a model
    lstm_model = LSTM.BiLSTM(input_dim=INPUT_DIM,
                             embedding_dim=params_d['embedding_dimension'],
                             hidden_dim=HIDDEN_DIM,
                             output_dim=params_d['output_dimension'],
                             n_layers=params_d['num_of_layers'],
                             dropout=DROPOUT,
                             input_rep=input_rep,
                             pad_idx=PAD_IDX)

    lstm_model.apply(_init_weights)

    # pretrained_embeddings = TEXT.vocab.vectors
    pretrained_embeddings = load_pretrained_embeddings(params_d['pretrained_embeddings_fn'], TEXT.vocab.itos)
    lstm_model.embedding.weight.data.copy_(pretrained_embeddings)
    # set pad tag embedding to 0
    lstm_model.embedding.weight.data[PAD_IDX] = torch.zeros(params_d['embedding_dimension'])

    model = {'lstm': lstm_model,
             'input_rep': input_rep,
             'TEXT': TEXT,
             'TAGS': UD_TAGS
             }

    return model


# no need for this one as part of the API
def get_model_params(model):
    """Returns a dictionary specifying the parameters of the specified model.
    This dictionary should be used to create another instance of the model.

    Args:
        model (torch.nn.Module): the network architecture

    Return:
        a dictionary, containing at least the following keys:
        {'input_dimension': int,
        'embedding_dimension': int,
        'num_of_layers': int,
        'output_dimension': int}
    """

    params_d = model.get_params_dict()
    return params_d


def load_pretrained_embeddings(path, vocab=None):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        You can also access the vectors at:
         https://www.dropbox.com/s/qxak38ybjom696y/glove.6B.100d.txt?dl=0
         (for efficiency (time and memory) - load only the vectors you need)
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.

    Args:
        path (str): full path to the embeddings file
        vocab (list): a list of words to have embeddings for. Defaults to None.

    """
    # make embeddings
    embeddings = torchtext.vocab.Vectors(path)
    if vocab is None:
        vectors = embeddings.vectors
    else:
        # make embedding vectors only for the vocabulary
        vectors = embeddings.get_vecs_by_tokens(vocab)
    return vectors


def train_rnn(model, train_data, val_data=None):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (dict): the model dict as returned by initialize_rnn_model()
        train_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus()
        val_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus() to be used for validation.
                            Defaults to None
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful
    """
    # Tips:
    # 1. you have to specify an optimizer
    # 2. you have to specify the loss function and the stopping criteria
    # 3. consider using batching
    # 4. some of the above could be implemented in helper functions (not part of
    #    the required API)

    BATCH_SIZE = 128

    # ## read data from dictionary
    lstm_model = model['lstm']
    input_rep = model['input_rep']
    TEXT = model['TEXT']
    UD_TAGS = model['TAGS']

    # ## split the data to sentences and tags
    x_train, y_train = _prepare_data(train_data)
    if input_rep == 0:
        # ## define torchtext fields
        fields = (("text", TEXT), ("udtags", UD_TAGS))
        # push the data into a torchtext type of dataset
        train_torchtext_dataset = LSTM.SequenceTaggingDataset([x_train, y_train], fields=fields)
        # ## create data iterator
        train_iterator = BucketIterator(
            train_torchtext_dataset,
            batch_size=BATCH_SIZE,
            device=device,
            # Function to use for sorting examples.
            sort_key=lambda x: len(x.text),
            # Sort all examples in data using `sort_key`.
            sort=False,
            # Shuffle data on each epoch run.
            shuffle=True,
            # Use `sort_key` to sort examples in each batch.
            sort_within_batch=True
        )

        # ## if validation data was sent, do the same for val_data
        if val_data is not None:
            x_val, y_val = _prepare_data(val_data)
            val_torchtext_dataset = LSTM.SequenceTaggingDataset([x_val, y_val], fields=fields)
            val_iterator = BucketIterator(
                val_torchtext_dataset,
                batch_size=BATCH_SIZE,
                device=device,
                sort_key=lambda x: len(x.text),
                sort=False,
                sort_within_batch=True
            )

    else:
        f_train, _ = _extract_words_features(x_train)
        FEATURES = Field(unk_token=None)
        FEATURES.build_vocab(f_train)
        lstm_model.set_features_field(FEATURES)
        fields = (("text", TEXT), ("udtags", UD_TAGS), ('features', FEATURES))
        train_torchtext_dataset = LSTM.SequenceTaggingDataset([x_train, y_train, f_train], fields=fields)
        train_iterator = BucketIterator(
            train_torchtext_dataset,
            batch_size=BATCH_SIZE,
            device=device,
            sort_key=lambda x: len(x.text),
            sort=False,
            shuffle=True,
            sort_within_batch=True
        )

        # ## if validation data was sent, do the same for val_data
        if val_data is not None:
            x_val, y_val = _prepare_data(val_data)
            f_val = _extract_words_features(x_val)
            val_torchtext_dataset = LSTM.SequenceTaggingDataset([x_val, y_val, f_val], fields=fields)
            val_iterator = BucketIterator(
                val_torchtext_dataset,
                batch_size=BATCH_SIZE,
                device=device,
                sort_key=lambda x: len(x.text),
                sort=False,
                sort_within_batch=True
            )

    optimizer = optim.Adam(lstm_model.parameters())
    TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)  # you can set the parameters as you like
    # vectors = load_pretrained_embeddings(pretrained_embeddings_fn)

    lstm_model = lstm_model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 17

    if val_data is None:
        _model_fit(lstm_model, train_iterator, optimizer, criterion, TAG_PAD_IDX, N_EPOCHS)
    else:
        _model_fit(lstm_model, train_iterator, optimizer, criterion, TAG_PAD_IDX, N_EPOCHS, val_iterator)


def rnn_tag_sentence(sentence, model):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence and t is the predicted tag.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict):  a dictionary with the trained BiLSTM model and all that is needed
                        to tag a sentence.

    Return:
        list: list of pairs
    """
    # ## extract relevant data from model dict
    lstm_model = model['lstm']
    input_rep = model['input_rep']
    TEXT = model['TEXT']
    UD_TAGS = model['TAGS']

    # ## put the model into evaluation mode
    lstm_model.eval()

    if TEXT.lower:
        tokens = [w.lower() for w in sentence]
    else:
        tokens = copy.copy(sentence)

    # ## numericalize the tokens using the vocabulary
    numericalized_tokens = [TEXT.vocab.stoi[w] for w in tokens]

    # ## find out which tokens are not in the vocabulary, i.e. are <unk> tokens
    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
    unks = [w for w, n in zip(tokens, numericalized_tokens) if n == unk_idx]

    # ## convert the numericalized tokens into a tensor and add a batch dimension
    token_tensor = torch.LongTensor(numericalized_tokens)
    token_tensor = token_tensor.unsqueeze(-1).to(device)

    # ## feed the tensor into the model
    if input_rep == 0:
        predictions = lstm_model(token_tensor)
    else:
        _, tensor_features = _extract_words_features(tokens)
        predictions = lstm_model(token_tensor, tensor_features)
    # ## get the predictions over the sentence
    top_predictions = predictions.argmax(-1)
    # ## convert the predictions into readable tags
    predicted_tags = [UD_TAGS.vocab.itos[p.item()] for p in top_predictions]

    # ## create tagged sentence
    tagged_sentence = []
    for word, tag in zip(sentence, predicted_tags):
        tagged_sentence.append((word, tag))

    return tagged_sentence


def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    model_params = {'max_vocab_size': -1,
                    'min_frequency': 2,
                    'input_rep': 1,
                    'embedding_dimension': 100,
                    'num_of_layers': 2,
                    'output_dimension': 18,
                    'pretrained_embeddings_fn': 'glove.6B.100d.txt',
                    'data_fn': 'en-ud-train.upos.tsv'
                    }

    return model_params


# ===========================================================
#       LSTM functions
# ===========================================================
def _prepare_data(dataset):
    """
    Gets labeled data (tuples) and return it a dictionary of x_data, y_data
    :param dataset:
    :return:
    """
    x_data = []
    labels = []
    for sent in dataset:
        sentence = []
        tags = []
        for word, tag in sent:
            sentence.append(word)
            tags.append(tag)
        x_data.append(sentence)
        labels.append(tags)

    return x_data, labels


def _init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)


def _epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / y[non_pad_elements].shape[0]


def _extract_words_features(sentences):
    """
    Returns a list of lists where every list is a list of size 3 which represent whether each word is lower-case,
    upper-case, or starts with a capital letter.
    :param sentences:
    :return: list.
    """
    features = []
    tensor_row = []
    lower = [1, 0, 0]
    upper = [0, 1, 0]
    leading = [0, 0, 1]
    other = [1, 1, 1]

    if type(sentences[0]) != list:
        for word in sentences:
            if word.islower():
                features.append('lower')
                tensor_row.append([lower])
            elif word.isupper():
                features.append('upper')
                tensor_row.append([upper])
            elif word[0].isupper():
                features.append('leading')
                tensor_row.append([leading])
            else:
                features.append('other')
                tensor_row.append([other])

    else:
        for sent in sentences:
            sent_features = []
            for word in sent:
                if word.islower():
                    sent_features.append('lower')
                elif word.isupper():
                    sent_features.append('upper')
                elif word[0].isupper():
                    sent_features.append('leading')
                else:
                    sent_features.append('other')
            features.append(sent_features)

    return features, torch.Tensor(tensor_row)


def _model_fit(model, train_iterator, optimizer, criterion, TAG_PAD_IDX, n_epochs=8, valid_iterator=None, verbose=1):
    """
    Fits the model to a given train set. Can be fitted with validation set as well to see accuracy and loss
    on validation throughout the training.
    :param model: a model object of type torch.NN
    :param train_iterator: of type BucketIterator. The train data (with labels)
    :param optimizer: The optimizer to use (from pytorch functions)
    :param criterion: The models criterion
    :param TAG_PAD_IDX: The index of padding
    :param n_epochs: Number of epochs to train on
    :param valid_iterator: of type BucketIterator. Validation data (with labels)
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

        train_loss, train_acc = _model_train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
        history['accuracy'].append(train_acc)
        history['loss'].append(train_loss)

        if valid_iterator is not None:
            valid_loss, valid_acc = _model_evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)
            history['val_accuracy'].append(valid_acc)
            history['val_loss'].append(valid_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = _epoch_time(start_time, end_time)

        if verbose == 1:
            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            if valid_iterator is not None:
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


def _model_train(model, iterator, optimizer, criterion, tag_pad_idx):
    """
    Trains the model on the train set and returns the epoch loss and acc
    :param model: a model object of type torch.NN
    :param iterator: of type BucketIterator. The train data
    :param optimizer: The optimizer to use (from pytorch functions)
    :param criterion: The models criterion
    :param tag_pad_idx: The index of padding
    :return: epoch loss, epoch acc
    """

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    # for sample_id, batch in enumerate(iterator.batches):
    for batch in iterator:
        text = batch.text
        tags = batch.udtags

        optimizer.zero_grad()

        # text = [sent len, batch size]
        if model.get_input_rep() == 0:
            predictions = model(text)
        else:
            features = batch.features
            predictions = model(text, features)

        # predictions = [sent len, batch size, output dim]
        # tags = [sent len, batch size]
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)

        # predictions = [sent len * batch size, output dim]
        # tags = [sent len * batch size]
        loss = criterion(predictions, tags)

        acc = _categorical_accuracy(predictions, tags, tag_pad_idx)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def _model_evaluate(model, iterator, criterion, tag_pad_idx):
    """
    Evaluate the model on a test/validation set and returns the epoch loss and acc
    :param model: a model object of type torch.NN
    :param iterator: of type BucketIterator. The evaluated data (with labels)
    :param criterion: The models criterion
    :param tag_pad_idx: The index of padding
    :return: epoch loss, epoch acc
    """
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

            acc = _categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# ===========================================================
#       Wrapper function (tagging with a specified model)
# ===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM)
           or the model isteld and the input_rep flag (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[model_dict]}
        4. BiLSTM+case: {'cblstm': [model_dict]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

        Parameters for an LSTM: the model dictionary (allows tagging the given sentence)


    Return:
        list: list of pairs
    """
    if list(model.keys())[0] == 'baseline':
        return baseline_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])
    if list(model.keys())[0] == 'hmm':
        return hmm_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])
    if list(model.keys())[0] == 'blstm':
        return rnn_tag_sentence(sentence, list(model.values())[0][0])
    if list(model.keys())[0] == 'cblstm':
        return rnn_tag_sentence(sentence, list(model.values())[0][0])


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence) == len(pred_sentence)

    correct = 0
    correctOOV = 0
    OOV = 0
    for i in range(len(gold_sentence)):
        if gold_sentence[i][1] == pred_sentence[i][1]:
            correct += 1
            if pred_sentence[i][0] not in VOCAB:
                correctOOV += 1
                OOV += 1
        elif pred_sentence[i][0] not in VOCAB:
            OOV += 1

    return correct, correctOOV, OOV
