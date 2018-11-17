#!/usr/bin/env python
# encoding: utf-8

"""
train_rerank.py

Created by Shuailong on 2017-01-13.

rerank training module for CCG Parsing.

"""

from __future__ import print_function
from __future__ import division
import numpy as np
from time import time


from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD


from model_rerank import LSTMReranking
from train import get_data
from train import get_embedding_matrix
from train import data_generator


np.random.seed(1337)  # for reproducibility

batch_size = 1
nb_epoch = 30

EMBEDDING_DIR = '../data'

MODEL_FILE = 'model_rerank.hdf5'
CANDIDATE_FILE = 'predictions.pickle'


def main():
    start_time = time()
    data = get_data()
    X_train = np.asarray(data['X_train'])
    X_dev = np.asarray(data['X_dev'])
    X_test = data['X_test']
    X_train_feats = np.asarray(data['X_train_feats'])
    X_dev_feats = np.asarray(data['X_dev_feats'])
    X_test_feats = data['X_test_feats']
    y_train = np.asarray(data['y_train'])
    y_dev = np.asarray(data['y_dev'])
    y_test = data['y_test']
    word_index = data['word_index']
    tag_index = data['tag_index']
    feature_sizes = data['feature_sizes']

    tag_size = len(tag_index)
    vocab_size = len(word_index)

    embedding_matrix = get_embedding_matrix(word_index)

    print('\nLoading candidate predictions for training data...')


    print('\nTraining reranking models...')
    model = LSTMReranking(vocab_size, feature_sizes, tag_size, embedding_matrix)
    model.summary()

    sgd = SGD(lr=0.01, momentum=0.7, clipnorm=5)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

    print('\nTrain...')
    checkpointer = ModelCheckpoint(MODEL_FILE,
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto')

    early_stopping = EarlyStopping(monitor='val_acc',
                                   min_delta=0,
                                   patience=3,
                                   verbose=1,
                                   mode='auto')

    model.fit_generator(data_generator(X_train, X_train_feats, y_train, tag_size, batch_size=batch_size),
                        samples_per_epoch=len(X_train)//batch_size*batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        callbacks=[checkpointer, early_stopping],
                        validation_data=data_generator(X_test, X_test_feats, y_test, tag_size, batch_size=batch_size),
                        nb_val_samples=len(X_test)//batch_size*batch_size
                        )
    model.load_weights(MODEL_FILE)

    print('\nTesting...')
    _, acc = model.evaluate_generator(data_generator(X_test, X_test_feats, y_test, tag_size),
                                      val_samples=len(X_test))

    print('Test accuracy: {}.'.format(acc))

    seconds = time() - start_time
    minutes = seconds / 60
    print('[Finished in {} seconds ({} minutes)]'.format(str(round(seconds, 1)),
                                                         str(round(minutes, 1))))

if __name__ == '__main__':
    main()
