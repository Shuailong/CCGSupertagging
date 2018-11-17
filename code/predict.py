#!/usr/bin/env python
# encoding: utf-8

"""
predict.py

Created by Shuailong on 2016-12-3.

Validate the correctness of the algorithm.

"""

from __future__ import print_function
from time import time
from keras.models import load_model
import os

from utils import true_accuracy
from utils import token2word
from dataset import get_data
from train import data_generator
from train import MODEL_FILE, MODEL_DIR


def main():
    start_time = time()
    print('\nGetting data...')
    data = get_data()
    X_test = data['X_test']
    X_test_feats = data['X_test_feats']
    y_test = data['y_test']
    tag_index = data['tag_index']
    tag_size = len(tag_index)
    word_index = data['word_index']

    index_word = {}
    for word, index in word_index.items():
        index_word[index] = word
    index_tag = {}
    for tag, index in tag_index.items():
        index_tag[index] = tag

    print('\nLoading model...')
    model = load_model(os.path.join(MODEL_DIR, MODEL_FILE), custom_objects={'true_accuracy': true_accuracy})

    print('\nPredicting...')

    samples = 1  # only 1 work for now
    prob = model.predict_generator(data_generator(X_test, X_test_feats, y_test, tag_size, shuffle=False),
                                   val_samples=samples)
    predict = prob.argmax(axis=-1)

    for i in range(samples):
        words = token2word(X_test[i], word_index)
        gold_tags = token2word(y_test[i], tag_index)
        tags = token2word(predict[i], tag_index)

        print('\n--------- Sample {}----------'.format(i))
        print('len(words): {} '.format(len(words),))

        assert len(words) == len(gold_tags) and len(words) == len(tags)
        print('Sentence:')
        print(words)
        print('Gold labeling:')
        print(gold_tags)
        print('Model labeling:')
        print(tags)

    seconds = time() - start_time
    minutes = seconds / 60
    print('[Finished in {} seconds ({} minutes)]'.format(str(round(seconds, 1)),
                                                         str(round(minutes, 1))))


if __name__ == '__main__':
    main()
