import re
import tagger as tagger
import time

# ===========================================
#       Load data
# ===========================================
start = time.time()

train_dataset = tagger.load_annotated_corpus('en-ud-train.upos.tsv')
test_dataset = tagger.load_annotated_corpus('en-ud-dev.upos.tsv')

# print(len(train_dataset))
# print(len(test_dataset))

x_test = []
y_test = []
for sent in test_dataset:
    sentence = []
    labels = []
    for word, tag in sent:
        sentence.append(word)
        labels.append(tag)
    x_test.append(sentence)
    y_test.append(labels)

# 0-allTagCounts, 1-perWordTagCounts, 2-transitionCounts, 3-emissionCounts, 4-A, 5-B
params = tagger.learn_params(train_dataset)
#
# allTagCount, perWordTagCounts, transitionCounts, emissionCounts, A, B = params

# #================= Baseline ================#
# acc = []
# correct_OOV = 0
# OOV_count = 0
# for sent, gold_sent in zip(x_test, test_dataset):
#     tagged_sentence = tagger.baseline_tag_sentence(sent, perWordTagCounts, allTagCount)
#     correct, correctOOV, OOV = tagger.count_correct(gold_sentence=gold_sent, pred_sentence=tagged_sentence)
#     acc.append(correct/len(sent))
#     correct_OOV += correctOOV
#     OOV_count += OOV
# print(f'base line accuracy: {sum(acc)/len(acc):.4f}')
# print(f'base line oov accuracy: {correct_OOV/OOV_count:.4f}')


# #================= HMM One sentence check ================#
# text = 'All of this started after their oil change .'
# sents = re.split(r'\s+', text)
# print(sents)
# tagged = tagger.hmm_tag_sentence(x_test[57], A, B)
# # print(tagged)
# # print(x_test[57])
# print(tagger.joint_prob(tagged, A, B))


# #================= HMM ================#
# acc = []
# correct_OOV = 0
# OOV_count = 0
# for sent, gold_sent in zip(x_test, test_dataset):
#     tagged_sentence = tagger.hmm_tag_sentence(sent, A, B)
#     correct, correctOOV, OOV = tagger.count_correct(gold_sentence=gold_sent, pred_sentence=tagged_sentence)
#     acc.append(correct/len(sent))
#     correct_OOV += correctOOV
#     OOV_count += OOV
# print(f'hmm accuracy: {sum(acc)/len(acc):.4f}')
# print(f'hmm oov accuracy: {correct_OOV/OOV_count:.4f}')
#
# end = time.time()
# print(f'time: {end-start:.4f}')


# #================= BiLSTM ================#

MIN_FREQ = 2
BATCH_SIZE = 128
# INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
# OUTPUT_DIM = len(UD_TAGS.vocab)
OUTPUT_DIM = 18
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
# PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

MAX_VOCAB = 20000

model_params = {'max_vocab_size': MAX_VOCAB,
            'min_frequency': MIN_FREQ,
            'input_rep': 0,
            'embedding_dimension': EMBEDDING_DIM,
            'num_of_layers': N_LAYERS,
            'output_dimension': OUTPUT_DIM,
            'pretrained_embeddings_fn': 'glove.6B.100d',
            'data_fn': 'en-ud-train.upos.tsv'
            }

model = tagger.initialize_rnn_model(model_params)
tagger.train_rnn(model, train_dataset)
# tag_sentence(sentence, model_to_use)



