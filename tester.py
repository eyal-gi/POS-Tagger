import re

import main as tagger
import time

# ===========================================
#       Load data
# ===========================================
start = time.time()

train_dataset = tagger.load_annotated_corpus('en-ud-train.upos.tsv')
test_dataset = tagger.load_annotated_corpus('en-ud-dev.upos.tsv')

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

allTagCount, perWordTagCounts, transitionCounts, emissionCounts, A, B = params

# #================= Baseline ================#
acc = []
for sent, gold_sent in zip(x_test, test_dataset):
    tagged_sentence = tagger.baseline_tag_sentence(sent, perWordTagCounts, allTagCount)
    correct, _, _ = tagger.count_correct(gold_sentence=gold_sent, pred_sentence=tagged_sentence)
    acc.append(correct/len(sent))
print(f'base line accuracy: {sum(acc)/len(acc):.4f}')


# #================= HMM One sentence check ================#
# text = 'All of this started after their oil change .'
# sents = re.split(r'\s+', text)
# print(sents)
# tagged = tagger.hmm_tag_sentence(x_test[57], A, B)
# # print(tagged)
# # print(x_test[57])
# print(tagger.joint_prob(tagged, A, B))



# #================= HMM ================#
acc = []
for sent, gold_sent in zip(x_test, test_dataset):
    tagged_sentence = tagger.hmm_tag_sentence(sent, A, B)
    correct, _, _ = tagger.count_correct(gold_sentence=gold_sent, pred_sentence=tagged_sentence)
    acc.append(correct/len(sent))
print(f'hmm accuracy: {sum(acc)/len(acc):.4f}')
end = time.time()
print(f'time: {end-start:.4f}')
