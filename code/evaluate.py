#!/usr/bin/env python
# encoding: utf-8

"""
evaluate.py

Created by Shuailong on 2016-12-2.

Evaluate model accuracy on test set.

"""

from __future__ import print_function

from time import time
from keras.models import load_model
import os

from utils import true_accuracy
from dataset import get_data
from train import MODEL_FILE, MODEL_DIR
from train import data_generator


def main():
    start_time = time()
    print('\nGetting data...')
    data = get_data(force=False)
    X_test = data['X_test']
    X_test_feats = data['X_test_feats']
    y_test = data['y_test']
    tag_size = len(data['tag_index'])

    print('\nLoading models...')
    model = load_model(os.path.join(MODEL_DIR, MODEL_FILE), custom_objects={'true_accuracy': true_accuracy})

    print('\nEvaluating...')
    _, true_acc = model.evaluate_generator(data_generator(X_test, X_test_feats, y_test, tag_size),
                                           val_samples=len(X_test))

    print('Test accuracy: {}.'.format(true_acc))
    seconds = time() - start_time
    minutes = seconds / 60
    print('[Finished in {} seconds ({} minutes)]'.format(str(round(seconds, 1)),
                                                         str(round(minutes, 1))))

if __name__ == '__main__':
    main()
