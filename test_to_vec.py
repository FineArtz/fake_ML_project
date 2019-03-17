# encoding=utf-8
# 2019-03-17
# Transfer test datas to vectors

import jieba
import codecs
import numpy as np
import pandas as pd

f = codecs.open("handout\\test_handout.txt", "r", encoding="utf-8")
commentList = []
strList = f.readlines()
for str in strList:
    commentList.append(str.strip())
f.close()
print(len(commentList))

ffout = codecs.open("washed_test_text.txt", "w")
stopList = [w.strip() for w in codecs.open("stop_words.txt", "r", encoding="utf-8").readlines()]
i = 0
for ss in commentList:
    segList = jieba.lcut(ss, cut_all=False)
    washedList = []
    for word in segList:
        word = word.strip()
        if word not in stopList:
            washedList.append(word + ("犇" if word != "" else ""))
    ffout.write(" ".join(washedList) + "\n")
    i = i + 1
ffout.close()

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

fin_d = codecs.open("washed_test_text.txt", "r")

ln = fin_d.readline()
fv = []
while ln != "":
    words = list(map(lambda s : s[0 : -1], ln.split(' ')))[0 : -1]
    vecs = getVec(words, model)
    if len(vecs) > 0:
        vec = sum(np.array(vecs)) / len(vecs)
    else:
        vec = np.zeros(300)
    fv.append(vec)
    ln = fin_d.readline()
fin_d.close()
print(len(fv))

X = np.array(fv)

data = pd.DataFrame(X)
data.to_csv("test_word_vector.csv")
