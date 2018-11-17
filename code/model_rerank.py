#!/usr/bin/env python
# encoding: utf-8

"""
model_rerank.py

Created by Shuailong on 2017-01-12.

Reranking Model for CCG Parsing.

"""

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Merge
from keras.regularizers import l2

EMBEDDING_DIM = 50
EXTRA_EMBEDDING_DIM = 32
TAG_EMBEDDING_DIM = 50


def FeatureEmbeddingLayer(token_size):
    '''
    Build prefix/suffix embedding layer.
    '''
    model = Sequential()
    model.add(Embedding(token_size+1,
                        EXTRA_EMBEDDING_DIM,
                        input_length=None,
                        weights=None,
                        trainable=True,
                        W_regularizer=l2(1e-6),
                        dropout=0.5))
    return model


def LSTMReranking(vocab_size, feature_sizes, tag_size, embedding_matrix):
    '''
    feature_sizes: [int], 1 to 4 character prefix/suffix token sizes
    Build the model for BiLSTM CCG tagging task.
    '''
    model_word = Sequential()
    model_word.add(Embedding(vocab_size+1,
                             EMBEDDING_DIM,
                             input_length=None,
                             weights=[embedding_matrix],
                             trainable=True,
                             W_regularizer=l2(1e-6),
                             dropout=0.5))
    model_cat = Sequential()
    model_cat.add(Embedding(tag_size+1,
                            TAG_EMBEDDING_DIM,
                            input_length=None,
                            weights=None,
                            trainable=True,
                            W_regularizer=l2(1e-6),
                            dropout=0.5))

    feature_layers = [FeatureEmbeddingLayer(size) for size in feature_sizes]

    layers = [model_word, model_cat] + feature_layers

    model = Sequential()
    model.add(Merge(layers, mode='concat', concat_axis=2))

    model.add(Bidirectional(LSTM(128, return_sequences=True, W_regularizer=l2(1e-6), U_regularizer=l2(1e-6), b_regularizer=l2(1e-6))))
    model.add(Bidirectional(LSTM(128, W_regularizer=l2(1e-6), U_regularizer=l2(1e-6), b_regularizer=l2(1e-6))))
    model.add(Dense(64, activation='relu', W_regularizer=l2(1e-6), b_regularizer=l2(1e-6)))
    model.add(Dense(tag_size, activation='softmax', W_regularizer=l2(1e-6), b_regularizer=l2(1e-6)))

    return model


def main():
    pass

if __name__ == '__main__':
    main()
