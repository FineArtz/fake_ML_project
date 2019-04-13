# encoding = utf-8
# 2019-04-12

from keras.models import Sequential, load_model
from keras import layers, losses, optimizers, metrics
import numpy as np 
import argparse
from util import *
import matplotlib.pyplot as plt 

def train(vecs_input, label_input, isTest = False):
    sp = [20, 300]

    model = Sequential()
    model.add(layers.InputLayer(input_shape=(sp[0], sp[1])))
    model.add(layers.SimpleRNN(units=128, activation="relu"))
    model.add(layers.Dense(units=1024, activation="sigmoid"))
    model.add(layers.Dense(units=512, activation="relu"))
    model.add(layers.Dense(units=2, activation="softmax"))

    l_r = 0.00005
    optm = optimizers.RMSprop(lr=l_r)
    model.compile(loss="categorical_crossentropy", optimizer=optm, metrics=["accuracy"])
    vs = 0.25 if isTest else 0
    eps = 20
    H = model.fit(x=vecs_input, y=label_input, batch_size=16, epochs=eps, validation_split=vs)
    if isTest:
        plt.figure()
        plt.plot(np.arange(eps), H.history["acc"])
        plt.plot(np.arange(eps), H.history["val_acc"])
        plt.plot(np.arange(eps), H.history['val_calcAUC'], "k-.")
        plt.show()
    
    model.save("Kmodel/keras-lstm-model")

def predict(vecs_test):
    model = load_model("Kmodel/keras-lstm-model")
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
