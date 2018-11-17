#!/usr/bin/env python
# encoding: utf-8

"""
train.py

Created by Shuailong on 2016-12-07.

training module for CCG Parsing.

"""


from __future__ import print_function
from __future__ import division
import numpy as np
import random
from time import time
import os

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import np_utils

from model import LSTMTagging
from dataset import get_data
from dataset import get_embedding_matrix
from utils import true_accuracy

np.random.seed(1337)  # for reproducibility

batch_size = 1
nb_epoch = 30

MODEL_FILE = 'model.best.hdf5'
MODEL_DIR = '../models'


def data_generator(X, X_feats, y, tag_size, batch_size=1, shuffle=True):
    '''
    Generate samples. Shuffles after one epoch.
    X: 2d array. First: samples; second: tokens.
    y: 2d array. First: samples; second: tags.
    X_feats: 3d array. First: feature_types; second: samples; third: feature tokens;
    tag_size: len(tags).

    return: [tokens, feat1_tokens, feat2_tokens, ..., feat8_tokens], y
    '''
    while True:
        orders = range(len(X))
        if shuffle:
            random.shuffle(orders)
        for i in range(len(X)//batch_size):
            Xs = []
            ys = []
            for j in range(i*batch_size, (i+1)*batch_size):
                Xs.append(X[orders[j]])
                ys.append(np_utils.to_categorical(y[orders[j]], tag_size))
            Xs = np.asarray(Xs).astype('int32')
            ys = np.asarray(ys).astype('int32')

            features = [None]*8
            for k in range(8):
                features[k] = []
                for j in range(i*batch_size, (i+1)*batch_size):
                    features[k].append(X_feats[k][orders[j]])
            features = [np.asarray(feature) for feature in features]
            yield [Xs] + features, ys


def main():
    start_time = time()
    data = get_data(force=True)
    X_train = data['X_train']
    X_dev = data['X_dev']
    X_test = data['X_test']
    X_train_feats = data['X_train_feats']
    X_dev_feats = data['X_dev_feats']
    X_test_feats = data['X_test_feats']
    y_train = data['y_train']
    y_dev = data['y_dev']
    y_test = data['y_test']
    word_index = data['word_index']
    tag_index = data['tag_index']
    feature_sizes = data['feature_sizes']
    extra_vocab = data['extra_vocab']

    tag_size = len(tag_index)

    embedding_matrix = get_embedding_matrix(word_index, extra_vocab, force=True)

    print('\nBuilding models...')
    model = LSTMTagging(feature_sizes, tag_size, embedding_matrix)
    model.summary()

    sgd = SGD(lr=0.01, momentum=0.7, clipnorm=5)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[true_accuracy])

    print('\nTrain...')
    checkpointer = ModelCheckpoint(os.path.join(MODEL_DIR, MODEL_FILE),
                                   monitor='val_true_accuracy',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto')

    early_stopping = EarlyStopping(monitor='val_true_accuracy',
                                   min_delta=0,
                                   patience=3,
                                   verbose=1,
                                   mode='auto')

    model.fit_generator(data_generator(X_train, X_train_feats, y_train, tag_size, batch_size=batch_size),
                        samples_per_epoch=len(X_train)//batch_size*batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        callbacks=[checkpointer, early_stopping],
                        validation_data=data_generator(X_dev, X_dev_feats, y_dev, tag_size, batch_size=batch_size),
                        nb_val_samples=len(X_dev)//batch_size*batch_size
                        )
    model.load_weights(os.path.join(MODEL_DIR, MODEL_FILE))

    print('\nTesting...')
    _, true_acc = model.evaluate_generator(data_generator(X_test, X_test_feats, y_test, tag_size),
                                           val_samples=len(X_test))

    print('Test accuracy: {}.'.format(true_acc))
    seconds = time() - start_time
    minutes = seconds / 60
    print('[Finished in {} seconds ({} minutes)]'.format(str(round(seconds, 1)),
                                                         str(round(minutes, 1))))

if __name__ == '__main__':
    main()
