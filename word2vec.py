# encoding = utf-8
# 2019-04-06
# word2vec using gensim

import codecs
from gensim.models.word2vec import Word2Vec

sentence = []

f = codecs.open("washed_text.txt", "r")
s = f.readline()
while s != "":
    words = list(map(lambda s : s.strip(), s.split(' ')))
    if words[-1] == "":
        words = words[0 : -1]
    sentence.append(words)
    s = f.readline()
f.close()

f = codecs.open("washed_test_text.txt", "r")
s = f.readline()
while s != "":
    words = list(map(lambda s : s.strip(), s.split(' ')))
    if words[-1] == "":
        words = words[0 : -1]
    sentence.append(words)
    s = f.readline()
f.close()

model = Word2Vec(min_count=1, size = 100)
model.build_vocab(sentence)
model.train(sentence, total_examples=model.corpus_count, epochs=model.iter)
model.save("word_mat.txt")
