#!/usr/bin/env python
# encoding: utf-8

"""
utils.py
Created by Shuailong on 2017-02-14.

"""

from __future__ import print_function
from __future__ import division

from keras.metrics import categorical_accuracy
from sklearn.model_selection import KFold


def extract_features(X):
    '''
    X: [[word11, word12, ...], [word21, word22, ...], ...]
    return: 1 to 4 prefix/suffix features.
    rtype: 3d array. First: feature types; second: samples; third: word pre/suffixes
    '''
    X_feats = []
    for sample in X:
        sample_features = []
        for word in sample:
            prefixes = [word[:i] for i in range(1, 5)]
            suffixes = [word[-i:] for i in range(1, 5)]
            sample_features.append(prefixes + suffixes)
        X_feats.append(sample_features)

    X_sep_feats = [None]*8
    for i in range(8):
        X_sep_feats[i] = []
        for sample_features in X_feats:
            sep_feat = []
            for word_feats in sample_features:
                sep_feat.append(word_feats[i])
            X_sep_feats[i].append(sep_feat)
    return X_sep_feats


def generate_ten_fold(X_train, X_train_feats, y_train):
    '''
    Generate 10-fold split.
    '''
    kf = KFold(n_splits=10, shuffle=False)
    for train, test in kf.split(X_train):
        yield X_train[train], X_train_feats[:, train], y_train[train], X_train[test], X_train_feats[:, test], y_train[test]


def true_accuracy(y_true, y_pred):
    '''
    Ignore START_OF_SENTENCE and END_OF_SENTENCE when calculating accuracy.
    Also ignore zero paddings.
    '''
    # ignore SOS, EOS
    trimmed_y_true = y_true[:, 1:-1, :]
    trimmed_y_pred = y_pred[:, 1:-1, :]

    return categorical_accuracy(trimmed_y_true, trimmed_y_pred)


def token2word(tokens, word_index):
    index_word = {v: k for k, v in word_index.items()}
    return [index_word[t] for t in tokens]


def word2token(words, word_index):
    pass
