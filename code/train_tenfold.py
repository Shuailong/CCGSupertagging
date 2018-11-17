#!/usr/bin/env python
# encoding: utf-8

"""
train_tenfold.py

Created by Shuailong on 2017-02-18.

training module for CCG Parsing.

"""

from __future__ import print_function
from __future__ import division
import numpy as np
import os
from time import time

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

from utils import generate_ten_fold
from utils import true_accuracy
from dataset import get_data
from dataset import get_embedding_matrix
from model import LSTMTagging
from train import data_generator
from train import MODEL_DIR


np.random.seed(1337)  # for reproducibility

batch_size = 1
nb_epoch = 50

MODEL_FILE = 'model_fold_{}.best.hdf5'


def main():
    start_time = time()
    data = get_data()
    X_train = np.asarray(data['X_train'])
    X_train_feats = np.asarray(data['X_train_feats'])
    y_train = np.asarray(data['y_train'])
    word_index = data['word_index']
    tag_index = data['tag_index']
    feature_sizes = data['feature_sizes']
    extra_vocab = data['extra_vocab']

    tag_size = len(tag_index)

    embedding_matrix = get_embedding_matrix(word_index, extra_vocab)

    tenfold_g = generate_ten_fold(X_train, X_train_feats, y_train)

    last_time = start_time

    for i, (x_train, x_train_feats, y_train, x_test, x_test_feats, y_test) in enumerate(tenfold_g):
        print('\nBuilding models for fold {}...'.format(i))
        model = LSTMTagging(feature_sizes, tag_size, embedding_matrix)
        # model.summary()

        sgd = SGD(lr=0.01, momentum=0.7, clipnorm=5)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[true_accuracy])

        print('\nTrain...')
        checkpointer = ModelCheckpoint(os.path.join(MODEL_DIR, MODEL_FILE.format(i)),
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

        model.fit_generator(data_generator(x_train, x_train_feats, y_train, tag_size, batch_size=batch_size),
                            samples_per_epoch=len(x_train)//batch_size*batch_size,
                            nb_epoch=nb_epoch,
                            verbose=1,
                            callbacks=[checkpointer, early_stopping],
                            validation_data=data_generator(x_test, x_test_feats, y_test, tag_size, batch_size=batch_size),
                            nb_val_samples=len(x_test)//batch_size*batch_size
                            )
        model.load_weights(os.path.join(MODEL_DIR, MODEL_FILE.format(i)))

        print('\nTesting...')
        _, true_acc = model.evaluate_generator(data_generator(x_test, x_test_feats, y_test, tag_size),
                                               val_samples=len(x_test))

        print('Test accuracy: {}.'.format(true_acc))

        time_per_fold = time() - last_time
        last_time = time()
        print('Time taken: {} s.'.format(time_per_fold))

    seconds = time() - start_time
    minutes = seconds / 60
    print('[Finished in {} seconds ({} minutes)]'.format(str(round(seconds, 1)),
                                                         str(round(minutes, 1))))

if __name__ == '__main__':
    main()
