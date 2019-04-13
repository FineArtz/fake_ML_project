# encoding = utf-8
# 2019-04-09

import codecs
from gensim.models.word2vec import Word2Vec
import numpy as np
import keras.backend as K
import tensorflow as tf

def word2vec(count = 1, dim = 100):
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

    model = Word2Vec(min_count=count, size=dim)
    model.build_vocab(sentence)
    model.train(sentence, total_examples=model.corpus_count, epochs=model.iter)
    model.save("word_mat.txt")

def genvec(inputFile):
    model = Word2Vec.load("word_mat.txt")

    def getVec(words, model):
        vecs = []
        for word in words:
            word = word.replace("\n", "")
            try:
                vecs.append(model[word])
            except KeyError:
                continue
        return np.array(vecs, dtype="float")

    fin_d = codecs.open(inputFile, "r")
    ln = fin_d.readline()
    fv = []
    maxlen = 0
    while ln != "":
        words = list(map(lambda s : s.strip(), ln.split(' ')))
        if words[-1] == "":
            words = words[0 : -1]
        maxlen = len(words) if maxlen < len(words) else maxlen
        vecs = np.array(getVec(words, model))
        assert(len(vecs) > 0)
        fv.append(vecs)
        ln = fin_d.readline()
    fin_d.close()

    if maxlen % 2 != 0:
        maxlen = maxlen + 1
        
    def normalize(vecs):
        sp = [vecs.shape[0], vecs.shape[1]]
        if sp[0] < maxlen:
            dif = maxlen - sp[0]
            vecs = vecs.flatten()
            vecs = np.concatenate((vecs, np.zeros(dif * sp[1], dtype=np.float)))
            vecs = vecs.reshape(-1, sp[1])
            assert(vecs.shape[0] == maxlen)
        return vecs

    return np.array(list(map(normalize, fv))) 

def genvec_weibo(inputFile):
    fin_m = codecs.open("bigram\\sgns.weibo.bigram", "r", encoding="utf-8")
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

    fin_d = codecs.open(inputFile, "r")
    ln = fin_d.readline()
    fv = []
    maxlen = 0
    while ln != "":
        words = list(map(lambda s : s.strip(), ln.split(' ')))
        if words[-1] == "":
            words = words[0 : -1]
        maxlen = len(words) if maxlen < len(words) else maxlen
        vecs = np.array(getVec(words, model))
        assert(len(vecs) > 0)
        fv.append(vecs)
        ln = fin_d.readline()
    fin_d.close()

    if maxlen % 2 != 0:
        maxlen = maxlen + 1
        
    def normalize(vecs):
        sp = [vecs.shape[0], vecs.shape[1]]
        if sp[0] < maxlen:
            dif = maxlen - sp[0]
            vecs = vecs.flatten()
            vecs = np.concatenate((vecs, np.zeros(dif * sp[1], dtype=np.float)))
            vecs = vecs.reshape(-1, sp[1])
            assert(vecs.shape[0] == maxlen)
        return vecs

    return np.array(list(map(normalize, fv))) 

def softmax(vec):
    s = sum(map(np.exp, vec))
    return list(map(lambda x : np.exp(x) / s, vec))

def randomSplit(x_data, y_data, test_size = 4000):
    assert(len(x_data) == len(y_data))
    total_size = len(x_data)
    test_id = np.random.choice(total_size, test_size, replace=False)
    test_id.sort()
    tmp = np.ones(total_size, dtype=bool)
    tmp[test_id] = False
    x_train = x_data[tmp]
    y_train = y_data[tmp]
    x_test = x_data[~tmp]
    y_test = y_data[~tmp]
    return x_train, y_train, x_test, y_test

def readLabel(file):
    f = codecs.open(file, "r")
    labels = []
    s = f.readline()
    while s != "":
        l = int(s.strip())
        labels.append([1, 0] if l == 0 else [0, 1])  
        s = f.readline()
    return np.array(labels)

def loadInput():
    vecs_input = genvec_weibo("washed_text.txt")
    label_input = readLabel("label.txt")
    return vecs_input, label_input

def loadTest():
    vecs_test = genvec_weibo("washed_test_text.txt")
    return vecs_test

def printResult(res):
    fout = codecs.open("handout.csv", "w")
    fout.write("id,pred\n")
    sz = len(res)
    for loop in range(sz):
        fout.write("%d,%.15f\n" % (loop, res[loop]))
    fout.close()

def calcAUC(y_true, y_pred, bins = 100):
    def PFA(y_true, y_pred, thr = K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= thr, "float32")
        N = K.sum(1 - y_true)
        FP = K.sum(y_pred - y_pred * y_true)
        return FP / N
    def PTA(y_true, y_pred, thr = K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= thr, "float32")
        P = K.sum(y_true)
        TP = K.sum(y_pred * y_true)
        return TP / P
    pfas = tf.stack([PFA(y_true, y_pred, k) for k in np.linspace(0, 1, bins)], axis=0)
    ptas = tf.stack([PTA(y_true, y_pred, k) for k in np.linspace(0, 1, bins)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    bsz = pfas[: -1] - pfas[1 :]
    s = ptas * bsz
    return K.sum(s, axis=0)

if __name__ == "__main__":
    y1 = np.array([0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7])
    y2 = np.array([1, 0, 0, 0, 1, 0, 1, 0])
    print("AUC = %.7lf" % calcAUC(y1, y2))
