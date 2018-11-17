#!/usr/bin/env python
# encoding: utf-8

"""
model.py

Created by Shuailong on 2016-11-29.

Model for CCG Parsing.

"""

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed, Merge
from keras.regularizers import l2

EMBEDDING_DIM = 50
FEAT_EMBEDDING_DIM = 32


def FeatureEmbeddingLayer(token_size):
    '''
    Build prefix/suffix embedding layer.
    '''
    model = Sequential()
    model.add(Embedding(token_size,
                        FEAT_EMBEDDING_DIM,
                        input_length=None,
                        weights=None,
                        trainable=True,
                        W_regularizer=l2(1e-6),
                        dropout=0.5))
    return model


def LSTMTagging(feature_sizes, tag_size, embedding_matrix):
    '''
    feature_sizes: [int], 1 to 4 character prefix/suffix token sizes
    Build the model for BiLSTM CCG tagging task.
    '''
    model_word = Sequential()
    model_word.add(Embedding(embedding_matrix.shape[0],
                             EMBEDDING_DIM,
                             input_length=None,
                             weights=[embedding_matrix],
                             trainable=True,
                             W_regularizer=l2(1e-6),
                             dropout=0.5))

    feature_layers = [FeatureEmbeddingLayer(size) for size in feature_sizes]

    layers = [model_word] + feature_layers

    model = Sequential()
    model.add(Merge(layers, mode='concat', concat_axis=2))

    model.add(Bidirectional(LSTM(128, return_sequences=True, W_regularizer=l2(1e-6), U_regularizer=l2(1e-6), b_regularizer=l2(1e-6))))
    model.add(Bidirectional(LSTM(128, return_sequences=True, W_regularizer=l2(1e-6), U_regularizer=l2(1e-6), b_regularizer=l2(1e-6))))
    model.add(TimeDistributed(Dense(64, activation='relu', W_regularizer=l2(1e-6), b_regularizer=l2(1e-6))))
    model.add(TimeDistributed(Dense(tag_size, activation='softmax', W_regularizer=l2(1e-6), b_regularizer=l2(1e-6))))

    return model


def main():
    pass

if __name__ == '__main__':
    main()
