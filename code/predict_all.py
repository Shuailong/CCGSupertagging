#!/usr/bin/env python
# encoding: utf-8

"""
predict.py

Created by Shuailong on 2017-01-13.

Generate candidate tags for the training data.

"""

from __future__ import print_function
from time import time
import numpy as np
import itertools
from six.moves import cPickle as pickle
import os

from keras.models import load_model
from keras.utils import generic_utils

from utils import true_accuracy
from utils import generate_ten_fold
from dataset import get_data
from dataset import CACHE_DIR
from train import data_generator
from train import MODEL_FILE, MODEL_DIR


PREDICTIONS_FILE = 'predictions.pickle'


def main():
    start_time = time()
    print('\nGetting data...')
    data = get_data()
    X_train = np.asarray(data['X_train'])
    X_train_feats = np.asarray(data['X_train_feats'])
    y_train = np.asarray(data['y_train'])
    tag_index = data['tag_index']
    tag_size = len(tag_index)

    beta = 0.6

    predictions = ''
    # prediction format
    # word gold_label label1 prob1 label2 prob2 ...

    for i, (x_train, x_train_feats, y_train, x_test, x_test_feats, y_test) \
            in enumerate(generate_ten_fold(X_train, X_train_feats, y_train)):
        # for each fold
        print('\nLoading models for fold {}...'.format(i))
        model = load_model(os.path.join(MODEL_DIR, MODEL_FILE.format(i)),
                           custom_objects={'true_accuracy': true_accuracy})

        print('\nPredicting...')
        progbar = generic_utils.Progbar(target=len(x_test))
        for j, (X, y) in enumerate(data_generator(x_test, x_test_feats, y_test,
                                                  tag_size, shuffle=False)):
            # for each sentence
            prob = model.predict_on_batch(X)
            prob = prob[0, :, :]  # len(sentence), 428
            maxes = prob.max(axis=1, keepdims=True)  # len(sentence), 1
            remains = prob > maxes*beta  # len(sentence), none
            s = ''
            for k, word in enumerate(remains):
                s += str(word) + ' ' + y[k]
                candidates = ''
                for idx, tag in enumerate(word):
                    if tag:
                        candidates += ' ' + idx + ' ' + prob[k][idx]
                s += candidates + '\n'
            # res = list(itertools.product(*tags))  # list of tuples

            predictions += s

            progbar.update(j)
            if j == len(x_test)-1:
                break
    # save prediction result to pickle file
    try:
        pickle.dump(predictions, open(os.path.join(CACHE_DIR, PREDICTIONS_FILE), 'wb'), pickle.HIGHEST_PROTOCOL)
    except Exception, e:
        raise e

    seconds = time() - start_time
    minutes = seconds / 60
    print('\n[Finished in {} seconds ({} minutes)]'.format(
        str(round(seconds, 1)), str(round(minutes, 1))))


if __name__ == '__main__':
    main()
