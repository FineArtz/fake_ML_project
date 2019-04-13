# encoding = utf-8
# 2019-04-12

# encoding = utf-8
# 2019-04-09

from keras import Model, layers, losses, optimizers, metrics, Input
from keras.models import load_model
from util import *
import codecs
import numpy as np 
import argparse
import matplotlib.pyplot as plt

def train(vecs_input, label_input, isTest = False):
    sp = [20, 300]

    x_input = Input(shape=(sp[0], sp[1]))
    conv_ker_size = [2, 3, 4, 5]
    convs = []
    for cks in conv_ker_size:
        conv = layers.Conv1D(filters=256, kernel_size=cks, strides=1, padding="same", activation="relu")(x_input)
        pool = layers.MaxPool1D(pool_size=sp[0] - cks + 1)(conv)
        pool = layers.Flatten()(pool)
        convs.append(pool)
    merge = layers.concatenate(convs, axis=1)
    output = layers.Dense(units=128, activation="relu")(merge)
    output = layers.Dropout(0.5)(output)
    output = layers.Dense(units=2, activation="softmax")(output)

    model = Model([x_input], output)
    l_r = 0.00005
    optm = optimizers.Adam(lr=l_r)
    model.compile(loss="categorical_crossentropy", optimizer=optm, metrics=["accuracy"])
    vs = 0.25 if isTest else 0
    eps = 6
    H = model.fit(x=vecs_input, y=label_input, batch_size=16, epochs=eps, validation_split=vs)
    if isTest:
        plt.figure()
        plt.plot(np.arange(eps), H.history["acc"])
        plt.plot(np.arange(eps), H.history["val_acc"])
        plt.plot(np.arange(eps), H.history['val_calcAUC'], "k-.")
        plt.show()
    
    model.save("Kmodel/keras-cnn-model")

def predict(vecs_test):
    model = load_model("Kmodel/keras-cnn-model")
    pred = model.predict(vecs_test, batch_size=1)
    
    result = []
    sz = vecs_test.shape[0]
    for loop in range(sz):
        result.append(pred[loop][1])
    return result

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-t", "--train", action="store_true")
    parse.add_argument("-p", "--pred", action="store_true")
    parse.add_argument("-c", "--crossval", action="store_true")
    parse.add_argument("--test", action="store_true")
    args = parse.parse_args()

    if args.train:
        v, l = loadInput()
        train(v, l)

    if args.pred:
        v = loadTest()
        r = predict(v)
        printResult(r)

    if args.test:
        v, l = loadInput()
        train(v, l, True)
