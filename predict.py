# encoding=utf-8
# 2019-03-17
# Make predictions

import codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
from sklearn.externals import joblib


clf = joblib.load("svm.model")

ddf = pd.read_csv("test_word_vector.csv")
XX = ddf.iloc[:, 1:]
xx_pca = PCA(n_components = 150).fit_transform(XX)

YY = clf.predict_proba(xx_pca)[:, 1]
print(YY)

fout = codecs.open("handout.csv", "w")
fout.write("id,pred\n")
i = 0
for pre in YY:
    fout.write("%d,%f\n" % (i, pre))
    i = i + 1
fout.close()
