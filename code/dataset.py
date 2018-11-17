#!/usr/bin/env python
# encoding: utf-8

"""
dataset.py
Created by Shuailong on 2016-11-24.
Dataset interface for CCG Parsing.
wsj00: dev
wsj23: test
wsj02-21: train
"""

from __future__ import print_function
from __future__ import division
import os
from keras import initializations
import numpy as np
from six.moves import cPickle as pickle

from preprocessing import Tokenizer
from preprocessing import UNKNOWN_UPPERCASE_ALNUM, UNKNOWN_LOWERCASE_ALNUM, UNKNOWN_NON_ALNUM
from preprocessing import START_OF_SENTENCE, END_OF_SENTENCE

from utils import token2word
from utils import extract_features
from model import EMBEDDING_DIM

DATA_DIR = '../data'
EMBEDDING_DIR = '../data/Turian'
CACHE_DIR = '../cache'


def get_data(force=False):
    picklefile = os.path.join(CACHE_DIR, 'data.pickle')
    if not force and os.path.isfile(picklefile):
        print('Loading data from pickle...')
        data = pickle.load(open(picklefile, 'rb'))
        return data

    print('\nLoading data...')
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = load_data()
    print(len(X_train), 'train sequences.')
    print(len(X_dev), 'dev sequences.')
    print(len(X_test), 'test sequences.')

    print('\nExtracting features...')
    X_train_sep_feats = extract_features(X_train)
    X_dev_sep_feats = extract_features(X_dev)
    X_test_sep_feats = extract_features(X_test)

    X_train_feat_tokens = []
    X_dev_feat_tokens = []
    X_test_feat_tokens = []

    feature_sizes = []
    for i in range(8):
        feat_tokenizer = Tokenizer(lower=False, cutoff=3, nb_unknowns=1, padding=True)
        feat_tokenizer.fit_on_texts(X_train_sep_feats[i])

        X_train_feat = feat_tokenizer.texts_to_sequences(X_train_sep_feats[i])
        X_dev_feat = feat_tokenizer.texts_to_sequences(X_dev_sep_feats[i])
        X_test_feat = feat_tokenizer.texts_to_sequences(X_test_sep_feats[i])

        X_train_feat_tokens.append(X_train_feat)
        X_dev_feat_tokens.append(X_dev_feat)
        X_test_feat_tokens.append(X_test_feat)

        feat_size = len(feat_tokenizer.word_index)
        feature_sizes.append(feat_size)

    # get dev and test vocabulary
    print('\nDev vocab:')
    dev_tokenizer = Tokenizer(lower=True, cutoff=0, nb_unknowns=3)
    dev_tokenizer.fit_on_texts(X_dev, verbose=True)
    print(len(dev_tokenizer.word_index.keys()))

    print('\nTest vocab:')
    test_tokenizer = Tokenizer(lower=True, cutoff=0, nb_unknowns=3)
    test_tokenizer.fit_on_texts(X_test, verbose=True)
    print(len(test_tokenizer.word_index.keys()))

    extra_vocab = set(dev_tokenizer.word_index.keys()+test_tokenizer.word_index.keys())
    print('\nTest/dev vocab: {}.'.format(len(extra_vocab)))

    print('\nTokenizing...')
    word_tokenizer = Tokenizer(lower=True, cutoff=0, nb_unknowns=3)
    word_tokenizer.fit_on_texts(X_train)
    X_train = word_tokenizer.texts_to_sequences(X_train)
    X_dev = word_tokenizer.texts_to_sequences(X_dev)
    X_test = word_tokenizer.texts_to_sequences(X_test)
    word_index = word_tokenizer.word_index

    tag_tokenizer = Tokenizer(lower=False, nb_words=425)
    tag_tokenizer.fit_on_texts(y_train)
    y_train = tag_tokenizer.texts_to_sequences(y_train)
    y_dev = tag_tokenizer.texts_to_sequences(y_dev)
    y_test = tag_tokenizer.texts_to_sequences(y_test)
    tag_index = tag_tokenizer.word_index
    print('Vocabulary size: {}. CCGTag size: {}.'.format(len(word_index), len(tag_index)))

    try:
        f = open(picklefile, 'wb')
        data = {
            'X_train': X_train,
            'X_test': X_test,
            'X_dev': X_dev,
            'y_train': y_train,
            'y_test': y_test,
            'y_dev': y_dev,
            'tag_index': tag_index,
            'word_index': word_index,
            'X_train_feats': X_train_feat_tokens,
            'X_dev_feats': X_dev_feat_tokens,
            'X_test_feats': X_test_feat_tokens,
            'feature_sizes': feature_sizes,
            'extra_vocab': extra_vocab
        }
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', picklefile, ':', e)
        raise

    return data


def get_embedding_matrix(word_index, extra_vocab, force=False):
    picklefile = os.path.join(CACHE_DIR, 'embedding_matrix.pickle')
    if not force and os.path.isfile(picklefile):
        print('Loading embedding matrix from pickle...')
        embedding_matrix = pickle.load(open(picklefile, 'rb'))
        return embedding_matrix

    print('\nLoading embeddings...')
    embeddings_index = {}
    with open(os.path.join(EMBEDDING_DIR, 'embeddings-scaled.50.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embeddings_index[START_OF_SENTENCE] = initializations.uniform(EMBEDDING_DIM, scale=2.0).eval()
    embeddings_index[END_OF_SENTENCE] = initializations.uniform(EMBEDDING_DIM, scale=2.0).eval()
    embeddings_index[UNKNOWN_UPPERCASE_ALNUM] = initializations.uniform(EMBEDDING_DIM, scale=2.0).eval()
    embeddings_index[UNKNOWN_LOWERCASE_ALNUM] = initializations.uniform(EMBEDDING_DIM, scale=2.0).eval()
    embeddings_index[UNKNOWN_NON_ALNUM] = initializations.uniform(EMBEDDING_DIM, scale=2.0).eval()

    print('\nFound {} word vectors.'.format(len(embeddings_index)-5))

    print('\nAdding dev/test vocab into word_index')
    # add dev and test vocabulary to word_index
    extra_vocab = list(set(embeddings_index.keys()) & extra_vocab)
    print('\nExtra vocab: {}.'.format(len(extra_vocab)))
    for word in extra_vocab:
        if word_index.get(word) is None:
            word_index[word] = len(word_index)
    print('\nCurrent vocab size: {}'.format(len(word_index)))

    oov = 0
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for i, word in enumerate(word_index):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        if '-' in word:
            embedding_vector = embeddings_index.get(word.split('-')[-1])
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue
        if '\/' in word:
            embedding_vector = embeddings_index.get(word.split('\/')[-1])
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                oov += 1
                embedding_matrix[i] = initializations.uniform(EMBEDDING_DIM, scale=2.0).eval()
    print('OOV number is {}. Total number is {}. Embedding OOV ratio is {}.'.format(oov, len(word_index), oov/len(word_index)))

    # save to pickle file
    try:
        f = open(picklefile, 'wb')
        pickle.dump(embedding_matrix, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', picklefile, ':', e)
        raise
    return embedding_matrix


def get_pairs(filename):
    '''
    return: [[word1, word2, ...], [word1, word2, ...]], [[stag1, stag2, ...], [stag1, stag2, ...]]
    '''

    X = []
    y = []
    with open(os.path.join(DATA_DIR, filename)) as f:
        for line in f:
            tokens = line.split(' ')
            X_seq = []
            y_seq = []
            for token in tokens:
                word, _, stag = token.split('|')
                X_seq.append(word)
                y_seq.append(stag)
            X.append(X_seq)
            y.append(y_seq)
    return X, y


def load_data():
    '''
    Load CCG supertagged data.
    '''
    X_train, y_train = get_pairs('wsj02-21.stagged')
    X_dev, y_dev = get_pairs('wsj00.stagged')
    X_test, y_test = get_pairs('wsj23.stagged')

    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)


if __name__ == '__main__':
    print('Dataset statistics:')
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = load_data()
    print('\n{} train sentences.'.format(len(X_train)))
    print('{} dev sentences.'.format(len(X_dev)))
    print('{} test sentences.'.format(len(X_test)))

    print('\nTrain sample:')
    print(X_train[0])
    print(y_train[0])
    print('\nDev sample:')
    print(X_dev[0])
    print(y_dev[0])
    print('\nTest sample:')
    print(X_test[0])
    print(y_test[0])

    max_train_len = max([len(x) for x in X_train])
    max_dev_len = max([len(x) for x in X_dev])
    max_test_len = max([len(x) for x in X_test])

    print('\nMax sentence length for train/dev/test: {}/{}/{}'.format(max_train_len, max_dev_len, max_test_len))

    print('\nTokenizing...')

    word_tokenizer = Tokenizer(lower=True, cutoff=0, nb_unknowns=3)
    word_tokenizer.fit_on_texts(X_train, verbose=True)

    tag_tokenizer = Tokenizer(lower=False, nb_words=425)
    tag_tokenizer.fit_on_texts(y_train, verbose=True)

    X_train_t = word_tokenizer.texts_to_sequences(X_train, verbose=True)
    X_dev_t = word_tokenizer.texts_to_sequences(X_dev, verbose=True)
    X_test_t = word_tokenizer.texts_to_sequences(X_test, verbose=True)

    y_train_t = tag_tokenizer.texts_to_sequences(y_train, verbose=True)
    y_dev_t = tag_tokenizer.texts_to_sequences(y_dev, verbose=True)
    y_test_t = tag_tokenizer.texts_to_sequences(y_test, verbose=True)

    print('\nTokenized train sample:')
    print(X_train_t[0])
    print(y_train_t[0])

    print('\nTokenized dev sample:')
    print(X_dev_t[0])
    print(y_dev_t[0])

    print('\nTokenized test sample:')
    print(X_test_t[0])
    print(y_test_t[0])

    word_index = word_tokenizer.word_index
    tag_index = tag_tokenizer.word_index

    # print('\nVocab list:')
    # for word in sorted(word_index, key=word_index.get):
    #     print('{}\t{}'.format(word, word_index[word]))
    # print('\nTag list:')
    # for tag in sorted(tag_index, key=tag_index.get):
    #     print('{}\t{}'.format(tag, tag_index[tag]))

    print('\nLoading embeddings...')
    embeddings_index = {}
    with open(os.path.join(EMBEDDING_DIR, 'embeddings-scaled.50.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found {} word vectors.'.format(len(embeddings_index)-5))
    embeddings_index[START_OF_SENTENCE] = initializations.uniform(50, scale=2.0).eval()
    embeddings_index[END_OF_SENTENCE] = initializations.uniform(50, scale=2.0).eval()
    embeddings_index[UNKNOWN_UPPERCASE_ALNUM] = initializations.uniform(50, scale=2.0).eval()
    embeddings_index[UNKNOWN_LOWERCASE_ALNUM] = initializations.uniform(50, scale=2.0).eval()
    embeddings_index[UNKNOWN_NON_ALNUM] = initializations.uniform(50, scale=2.0).eval()

    # add dev and test vocabulary into word_index
    print('\nDev vocab:')
    dev_tokenizer = Tokenizer(lower=True, cutoff=0, nb_unknowns=3)
    dev_tokenizer.fit_on_texts(X_dev, verbose=True)
    print(len(dev_tokenizer.word_index.keys()))

    print('\nTest vocab:')
    test_tokenizer = Tokenizer(lower=True, cutoff=0, nb_unknowns=3)
    test_tokenizer.fit_on_texts(X_test, verbose=True)
    print(len(test_tokenizer.word_index.keys()))

    total_vocab = set(dev_tokenizer.word_index.keys()+test_tokenizer.word_index.keys())
    print('\nTotal vocab: {}.'.format(len(total_vocab)))
    valid_vocab = list(set(embeddings_index.keys()) & total_vocab)
    print('\nValid vocab: {}.'.format(len(valid_vocab)))
    for word in valid_vocab:
        if word_index.get(word) is None:
            word_index[word] = len(word_index)
    print('\nCurrent vocab size: {}'.format(len(word_index)))

    oov = 0
    embedding_matrix = np.zeros((len(word_index) + 1, 50))
    for i, word in enumerate(word_index):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        if '-' in word:
            embedding_vector = embeddings_index.get(word.split('-')[-1])
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue
        if '\/' in word:
            embedding_vector = embeddings_index.get(word.split('\/')[-1])
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                oov += 1
                print(word)
                embedding_matrix[i] = initializations.uniform(50, scale=2.0).eval()
    print('OOV number is {}. Total number is {}. Embedding OOV ratio is {}.'.format(oov, len(word_index), oov/len(word_index)))

    print('\nExtracting features...')
    X_train_sep_feats = extract_features(X_train)
    X_dev_sep_feats = extract_features(X_dev)
    X_test_sep_feats = extract_features(X_test)

    X_train_feat_tokens = []
    X_dev_feat_tokens = []
    X_test_feat_tokens = []

    feature_sizes = []
    for i in range(8):
        feat_tokenizer = Tokenizer(lower=False, cutoff=3, nb_unknowns=1, padding=True)
        feat_tokenizer.fit_on_texts(X_train_sep_feats[i], verbose=True)

        X_train_feat = feat_tokenizer.texts_to_sequences(X_train_sep_feats[i])
        X_dev_feat = feat_tokenizer.texts_to_sequences(X_dev_sep_feats[i])
        X_test_feat = feat_tokenizer.texts_to_sequences(X_test_sep_feats[i])

        X_train_feat_tokens.append(X_train_feat)
        X_dev_feat_tokens.append(X_dev_feat)
        X_test_feat_tokens.append(X_test_feat)

        feat_size = len(feat_tokenizer.word_index)
        feature_sizes.append(feat_size)
    print('\nFeature sizes: {}'.format(feature_sizes))
    # print('\nFeature vocab: ')
    # print(feat_tokenizer.word_index.keys())
    print('\nFeature sample:')
    print(X_train_sep_feats[7][0])
    tokens = X_train_feat_tokens[7][0]
    print(token2word(tokens, feat_tokenizer.word_index))
    print('\nOriginal sentence:')
    print(X_train[0])
