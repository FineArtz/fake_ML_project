# encoding = utf-8
# 2019-04-09

import tensorflow as tf 
from util import *
import codecs
import numpy as np 
import argparse
import matplotlib.pyplot as plt

def train(vecs_input, label_input, isCV = False, vecs_test = None, label_test = None):
    istrain = True
    isTest = False if vecs_test is None else True

    sp = [20, 300]
    x_input = tf.placeholder(tf.float32, [None, sp[0] * sp[1]])
    y_output = tf.placeholder(tf.float32, [None, 2])
    x_input_vecs = tf.reshape(x_input, [-1, sp[0], sp[1]])

    conv1 = tf.layers.conv1d(inputs=x_input_vecs, filters=32, kernel_size=7, strides=1, padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)
    conv2 = tf.layers.conv1d(inputs=pool1, filters=64, kernel_size=7, strides=1, padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)
    flat = tf.reshape(pool2, [-1, int(sp[0] * 64 / 4)])
    dense = tf.layers.dense(inputs=flat, units=128, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=istrain)
    output = tf.layers.dense(inputs=dropout, units=2)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_output, logits=output))
    optm = tf.train.AdamOptimizer(learning_rate=0.00064).minimize(loss)
    accuracy = tf.metrics.accuracy(labels=tf.argmax(y_output, axis=1), predictions=tf.argmax(output, axis=1))[1]

    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    saver = tf.train.Saver(max_to_keep=1)
    batch_size = 16

    best_acc = 0
    best_loss = 1
    sz = len(vecs_input)

    auc_list = [] if isTest else None
    loop = -1

    for i in range(16000 // 16 * 2):
        loop = (loop + 1) % (sz // batch_size)
        batch = vecs_input[loop * batch_size : (loop + 1) * batch_size].reshape(-1, 6000)
        label = label_input[loop * batch_size : (loop + 1) * batch_size].reshape(-1, 2)
        train_loss, train_op = sess.run([loss, optm], feed_dict={x_input:batch, y_output:label})
        test_accuracy = sess.run(accuracy, feed_dict={x_input:batch, y_output:label})

        if test_accuracy - best_acc > -1e-4:
            if best_loss > train_loss:
                best_acc = test_accuracy
                best_loss = train_loss
                if isCV:
                    saver.save(sess, "tfmodeltmp/tf-model", global_step=None)
                else:
                    saver.save(sess, "tfmodel/tf-model", global_step=None)
        if loop % 50 == 0:
            print("Step = %d, TrainLoss = %.4f, Test accuracy = %.2f" % (loop, train_loss, test_accuracy))
            if isTest:
                szt = vecs_test.shape[0]
                result = []
                for ll in range(szt):
                    batch = vecs_test[ll].reshape(-1, 6000)
                    test_output = sess.run(output, feed_dict={x_input:batch})
                    prd = softmax([test_output[0][0], test_output[0][1]])
                    result.append(prd[1])
                y = np.array([x[1] for x in label_test])
                auc = calcAUC(y, result)
                auc_list.append(auc)
                print("AUC = %.7lf" % auc)

    sess.close()
    if isTest:
        ssz = len(auc_list)
        plt.plot(np.arange(ssz) * 50, np.array(auc_list))
        plt.show()

def predict(vecs_test, isCV = False):
    tf.reset_default_graph()
    istrain = False

    sp = [20, 300]
    x_input = tf.placeholder(tf.float32, [None, sp[0] * sp[1]])
    y_output = tf.placeholder(tf.float32, [None, 2])
    x_input_vecs = tf.reshape(x_input, [-1, sp[0], sp[1]])

    conv1 = tf.layers.conv1d(inputs=x_input_vecs, filters=32, kernel_size=7, strides=1, padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)
    conv2 = tf.layers.conv1d(inputs=pool1, filters=64, kernel_size=7, strides=1, padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)
    flat = tf.reshape(pool2, [-1, int(sp[0] * 64 / 4)])
    dense = tf.layers.dense(inputs=flat, units=128, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=istrain)
    output = tf.layers.dense(inputs=dropout, units=2)

    sess = tf.Session()
    saver = tf.train.Saver()
    if isCV:
        saver.restore(sess, "tfmodeltmp/tf-model")
    else:
        saver.restore(sess, "tfmodel/tf-model")
    
    result = []
    sz = vecs_test.shape[0]
    for loop in range(sz):
        batch = vecs_test[loop].reshape(-1, 6000)
        test_output = sess.run(output, feed_dict={x_input:batch})
        prd = softmax([test_output[0][0], test_output[0][1]])
        result.append(prd[1])
    sess.close()
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

    if args.crossval:
        v, l = loadInput()
        x_train, y_train, x_test, y_test = randomSplit(v, l)
        train(x_train, y_train, True)
        r = predict(x_test, True)
        y = np.array([x[1] for x in y_test])
        auc = calcAUC(r, y)
        print("AUC = %.7lf\n" % auc)

    if args.test:
        v, l = loadInput()
        x_train, y_train, x_test, y_test = randomSplit(v, l)
        train(x_train, y_train, True, x_test, y_test)
