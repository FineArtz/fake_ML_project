# encoding=utf-8
# 2019-03-17
# Train SVM

import codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
from sklearn.externals import joblib

df = pd.read_csv("word_vector2.csv")
y = df.iloc[:, 1]
x = df.iloc[:, 2:]
print(x)
x_pca = PCA(n_components = 150).fit_transform(x)

clf = svm.SVC(C = 0.8, kernel = "rbf", gamma = 5.0, probability = True)
clf.fit(x_pca,y)

joblib.dump(clf, "svm.model")
clf = joblib.load("svm.model")

print('Test Accuracy: %.2f'% clf.score(x_pca,y))
pred_probas = clf.predict_proba(x_pca)[:, 1] #score
print(pred_probas)

fpr, tpr, _ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = 'area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')
plt.show()

dff = pd.read_csv("test_word_vector.csv")
XX = dff.iloc[:, 1:]
xx_pca = PCA(n_components = 150).fit_transform(XX)
print(xx_pca)
YY = clf.predict_proba(xx_pca)[:, 1]

fout = codecs.open("handout.csv", "w")
fout.write("id,pred\n")
i = 0
for pre in YY:
    fout.write("%d,%f\n" % (i, pre))
    i = i + 1
fout.close()
