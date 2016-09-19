#!/usr/bin/python

import collections
import math
import os
import random

import numpy as np
import tensorflow as tf
from random import shuffle

import sys, getopt
import os

from collections import namedtuple
import nltk
from nltk import word_tokenize

Dataset = namedtuple('Dataset','sentences labels')
random.seed(2)
num_classes = 3
learning_rate = 0.05
num_epochs = 2
embedding_dim = 10
label_to_id = {'World':0, 'Entertainment':1, 'Sports':2}
unknown_word_id = 0
word_to_id = dict()
batch_size = 200

def create_label_vec(label):
   # Generate a label vector for a given classification label.
   temp = [0,0,0]
   temp[label_to_id[label[0]]] = 1
   return temp

def tokenize(sens):
    # Tokenize a given sentence into a sequence of tokens.
    return word_tokenize(sens)

def map_token_seq_to_word_id_seq(token_seq, word_to_id):
    return [map_word_to_id(word_to_id,word) for word in token_seq]


def map_word_to_id(word_to_id, word):
    # map each word to its id.
    if word in word_to_id:
        return (word_to_id[word])
    else:
        return (word_to_id['$UNK$'])


def build_vocab(sens_file_name):
    data = []
    with open(sens_file_name) as f:
        for line in f.readlines():
            tokens = tokenize(line)
            data.extend(tokens)
    count = [['$UNK$', 0]]
    sorted_counts = collections.Counter(data).most_common()
    count.extend(sorted_counts)
    word_to_id = dict()
    for word, _ in count:
        word_to_id[word] = len(word_to_id)
    print('size of vocabulary is %s. ' % len(word_to_id))
    return word_to_id


def read_labeled_dataset(sens_file_name, label_file_name, word_to_id):
    sens_file = open(sens_file_name)
    label_file = open(label_file_name)
    data = []
    for label in label_file:
        #print(label)
        sens = sens_file.readline()
        word_id_seq = map_token_seq_to_word_id_seq(tokenize(sens), word_to_id)
        data.append((word_id_seq, create_label_vec(tokenize(label))))
    print("read %d sentences from %s ." % (len(data), sens_file_name))
    sens_file.close()
    label_file.close()
    return data

def read_dataset(sens_file_name, word_to_id):
    sens_file = open(sens_file_name)
    data = []
    for sens in sens_file:
        word_id_seq = map_token_seq_to_word_id_seq(tokenize(sens), word_to_id)
        data.append(word_id_seq)
    print("read %d sentences from %s ." % (len(data), sens_file_name))
    sens_file.close()
    return data


def eval(word_to_id, train_dataset, dev_dataset, test_dataset):
    num_words = len(word_to_id)
    # Initialize the placeholders and Variables. E.g.
    correct_label = tf.placeholder(tf.float32, shape=[num_classes])
    #correct_label = tf.placeholder(tf.float32, shape=[num_classes])
    #input_sens = tf.placeholder(tf.float32, shape = [None, len(word_to_id)])
    # Hint: use [None] when you are not certain about the value of shape
    test_results = []
    #print("here")
    #x = tf.placeholder(tf.float32, [None, len(word_to_id)])
    input_sens = tf.placeholder(tf.int32, shape=[None])
    embeddings = tf.Variable(tf.random_uniform([num_words, embedding_dim], -1.0, 1.0))
    new_mat = tf.Variable(tf.random_uniform([embedding_dim, num_classes], -1.0, 1.0))
    #W = tf.Variable(tf.zeros([len(word_to_id), num_classes]))
    #W = tf.Variable(tf.zeros([len(word_to_id), num_classes]))
    #b = tf.Variable(tf.zeros([num_classes]))
    #y = tf.nn.softmax(tf.matmul(x, W) + b)
    #print("here!")
    with tf.Session() as sess:
        # Write code for constructing computation graph here.
        # Hint:
        #    1. Find the math operations at https://www.tensorflow.org/versions/r0.10/api_docs/python/math_ops.html
        #    2. Try to reuse/modify the code from tensorflow tutorial.
        #    3. Use tf.reshape if the shape information of a tensor gets lost during the contruction of computation graph.

        embed = tf.nn.embedding_lookup(embeddings, input_sens)
        tmp_m = tf.reduce_sum(embed, 0)
        sum_rep = tf.reshape(tmp_m, [1, embedding_dim])
        #evaluation code, assume y is the estimated probability vector of each class
        y = tf.nn.softmax(tf.matmul(sum_rep, new_mat))

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(correct_label, 0))
        accuracy = tf.cast(correct_prediction, tf.float32)

        #correct_label = tf.placeholder(tf.float32, [None, num_classes])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(correct_label * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        prediction = tf.cast(tf.argmax(y, 1), tf.int32)

        sess.run(tf.initialize_all_variables())
        # In this assignment it is sufficient to use GradientDescentOptimizer, you are not required to implement a regularizer.

        #print("here!!")
        for epoch in range(num_epochs):
            shuffle(train_dataset)
            # Writing the code for training. It is not required to use a batch with size larger than one.
            for (sens, label) in train_dataset:
                train_step.run({input_sens: sens, correct_label:label})
            print('Epoch %d : %s .' % (epoch,compute_accuracy(accuracy, input_sens, correct_label, dev_dataset)))
        # uncomment the following line in the grading lab for evaluation
        print('Accuracy on the test set : %s.' % compute_accuracy(accuracy, input_sens, correct_label, test_dataset))
        # input_sens is the placeholder of an input sentence.
        test_results = predict(prediction, input_sens, test_dataset)
    return test_results


def compute_accuracy(accuracy, input_sens, correct_label, eval_dataset):
    num_correct = 0
    for (sens, label) in eval_dataset:
        #num_correct += accuracy.eval(feed_dict={input_sens: X, correct_label: Y})
        num_correct += accuracy.eval(feed_dict={input_sens: sens, correct_label: label})
        #if num_correct % 1000 == 0:
        #    print(num_correct)
    print('#correct sentences is %s ' % num_correct)
    return num_correct / len(eval_dataset)


def predict(prediction, input_sens, test_dataset):
    test_results = []
    for (sens, label) in test_dataset:
        test_results.append(prediction.eval(feed_dict={input_sens: sens}))
    return test_results


def write_result_file(test_results, result_file):
    with open(result_file, mode='w') as f:
         for r in test_results:
             f.write("%d\n" % r)


def main(argv):
    trainSensFile = ''
    trainLabelFile = ''
    devSensFile = ''
    devLabelFile = ''
    testSensFile = ''
    testLabelFile = ''
    testResultFile = ''
    try:
        opts, args = getopt.getopt(argv,"hd:",["dataFolder="])
    except getopt.GetoptError:
        print('fastText.py -d <dataFolder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('fastText.py -d <dataFolder>')
            sys.exit()
        elif opt in ("-d", "--dataFolder"):
            trainSensFile = os.path.join(arg, 'sentences_train.txt')
            devSensFile = os.path.join(arg, 'sentences_dev.txt')
            testSensFile = os.path.join(arg, 'sentences_test.txt')
            trainLabelFile = os.path.join(arg, 'labels_train.txt')
            devLabelFile = os.path.join(arg, 'labels_dev.txt')
            ## uncomment the following line in the grading lab
            testLabelFile = os.path.join(arg, 'labels_test.txt')
            testResultFile = os.path.join(arg, 'test_results.txt')
        else:
            print("unknown option %s ." % opt)
    ## Please write the main procedure here by calling appropriate methods.
    word_to_id = build_vocab(trainSensFile)
    #word_to_id = word_to_id[:1000]
    #print(read_labeled_dataset(trainSensFile, trainLabelFile, word_to_id))
    #print(word_to_id['Radical'])
    train_dataset = read_labeled_dataset(trainSensFile, trainLabelFile, word_to_id)
    #print(train_dataset[:10])
    dev_dataset = read_labeled_dataset(devSensFile, devLabelFile, word_to_id)
    #print(dev_dataset[:10])
    test_dataset = read_labeled_dataset(testSensFile, testLabelFile, word_to_id)
    #test_dataset = read_dataset(testSensFile, word_to_id)
    test_results = eval(word_to_id, train_dataset, dev_dataset, test_dataset)
    write_result_file(test_results, testResultFile)

if __name__ == "__main__":
   main(sys.argv[1:])
