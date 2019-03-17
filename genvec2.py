# encoding=utf-8
# 2019-03-17
# Generate word vectors using word2vec

import codecs
import numpy as np
import pandas as pd

fin_m = codecs.open("bigram\\sgns.wiki.bigram", "r", encoding="utf-8")
model = {}
ln = fin_m.readline()
ln = fin_m.readline()
while ln != "":
    words = ln.split(' ')
    model[words[0]] = list(map(float, words[1 : -1]))
    ln = fin_m.readline()
fin_m.close()

def getVec(words, model):
    vecs = []
    for word in words:
        word = word.replace("\n", "")
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype="float")

fin_p = codecs.open("positive.txt", "r", encoding="utf-8")
fin_n = codecs.open("negative.txt", "r")

ln = fin_p.readline()
fvp = []
while ln != "":
    words = list(map(lambda s : s[0 : -1], ln.split(' ')))[0 : -1]
    vecs = getVec(words, model)
    if len(vecs) > 0:
        vec = sum(np.array(vecs)) / len(vecs)
        fvp.append(vec)
    ln = fin_p.readline()

ln = fin_n.readline()
fvn = []
while ln != "":
    words = list(map(lambda s : s[0 : -1], ln.split(' ')))[0 : -1]
    vecs = getVec(words, model)
    if len(vecs) > 0:
        vec = sum(np.array(vecs)) / len(vecs)
        fvn.append(vec)
    ln = fin_n.readline()

fin_p.close()
fin_n.close()

Y = np.concatenate((np.ones(len(fvp)), np.zeros(len(fvn))))
X = fvp
for n in fvn:
    X.append(n)
X = np.array(X)

df_x = pd.DataFrame(X)
df_y = pd.DataFrame(Y)
data = pd.concat([df_y, df_x], axis=1)
data.to_csv("word_vector2.csv")
