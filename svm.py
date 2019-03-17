#!/usr/bin/env python
# -*- coding: utf-8  -*-

import codecs
import numpy as np
from sklearn import svm
from sklearn.model_selection import *
from sklearn import metrics
import matplotlib.pyplot as plt

fin = codecs.open("word_vector.csv", "r")
lines = fin.readlines()
x = np.array([])
y = np.array([])
for ln in lines:
    datas = np.array(list(map(float, ln.split(','))))
    y = np.append(y, int(datas[0]))
    x = np.append(x, datas[1 : 101])
fin.close()
x = np.reshape(x, (-1, 100))
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 1)
clf = svm.SVC(C = 2, probability=True)
clf.fit(x_train, y_train.ravel())

print(clf.score(x_train, y_train))
print(clf.score(x_test, y_test))

pred_probas = clf.predict_proba(x)[:,1] #score

fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = 'area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')
plt.show()
