#!/usr/bin/env python
# encoding: utf-8
'''
Text preprocessing module modified by Shuailong.
https://github.com/fchollet/keras/blob/master/keras/preprocessing/text.py
'''

from __future__ import absolute_import
from __future__ import division

import numpy as np
from six.moves import range
from six.moves import zip
import re


UNKNOWN = '*UNKNOWN*' # same with Turian 10
UNKNOWN_UPPERCASE_ALNUM = '*UNKNOWN_UPPERCASE_ALNUM*'
UNKNOWN_LOWERCASE_ALNUM = '*UNKNOWN_LOWERCASE_ALNUM*'
UNKNOWN_NON_ALNUM = '*UNKNOWN_NON_ALNUM*'

START_OF_SENTENCE = '*START_OF_SENTENCE*'
END_OF_SENTENCE = '*END_OF_SENTENCE*'


class Tokenizer(object):
    def __init__(self, nb_words=None, lower=False, cutoff=0, nb_unknowns=1, padding=True):
        '''The class allows to vectorize a text corpus, by turning each
        text into either a sequence of integers (each integer being the index
        of a token in a dictionary) or into a vector where the coefficient
        for each token could be binary, based on word count, based on tf-idf...
        # Arguments
            nb_words: the maximum number of words to keep, based
                on word frequency. Only the most common `nb_words` words will
                be kept.
            lower: boolean. Whether to convert the texts to lowercase.
            cutoff: when building the vocabulary ignore terms that have a frequency strictly lower than the given value.
            nb_unknowns: 1 or 3. default to 1. if 3, distinguish between unpper case alnum, lower case alnum, and nonalnum unknowns.
        `0` is a reserved index that won't be assigned to any word.
        '''
        self.word_counts = {}
        self.lower = lower
        self.nb_words = nb_words
        self.cutoff = cutoff
        self.nb_unknowns = nb_unknowns
        self.padding = padding

    def fit_on_texts(self, texts, verbose=False):
        '''Required before using texts_to_sequences or texts_to_matrix
        # Arguments
            texts: a list of word sequences
        '''
        for seq in texts:
            for w in seq:
                w = w.strip()
                if self.lower:
                    w = w.lower()
                w = re.sub('\d', '0', w)
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
        if self.cutoff > 0:
            self.word_counts = {k:v for k, v in self.word_counts.items() if v >= self.cutoff}

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts][:self.nb_words]
        self.word_index = dict(list(zip(sorted_voc, list(range(len(sorted_voc))))))
        vocab_size = len(self.word_index)
        if self.padding:  
            self.word_index[START_OF_SENTENCE] = len(self.word_index)
            self.word_index[END_OF_SENTENCE] = len(self.word_index)

        if self.nb_unknowns == 1:
            self.word_index[UNKNOWN] = len(self.word_index)
        else:
            # self.nb_unknowns == 3:
            self.word_index[UNKNOWN_UPPERCASE_ALNUM] = len(self.word_index)
            self.word_index[UNKNOWN_LOWERCASE_ALNUM] = len(self.word_index)
            self.word_index[UNKNOWN_NON_ALNUM] = len(self.word_index)

        nb_paddings = 2 if self.padding else 0
        if verbose:
            print('Original vocabulary size: {}'.format(vocab_size))
            print('Total vocabulary size: {} (including {} UNKNOWN types, {} padding types)'.format(
                len(self.word_index), self.nb_unknowns, nb_paddings))

    def texts_to_sequences(self, texts, verbose=False):
        '''Transforms each text in texts in a sequence of integers.
        Only top "nb_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        Returns a list of sequences.
        '''
        res = []
        total_len = 0
        total_unknowns = []
        for vect, unknowns, sent_len in self.texts_to_sequences_generator(texts):
            res.append(vect)
            total_len += sent_len
            total_unknowns += unknowns
        if verbose:
            from collections import Counter
            c = Counter(total_unknowns)
            if len(c) > 0:
                max_f = c[max(c, key=c.get)]
            else:
                max_f = 0
            print('Total unknows tokens: {}. Max frequency: {}.'.format(len(c.keys()), max_f))
            print('Total unknown freq: {}. Total tokens: {}. Ratio: {}'.format(
                len(total_unknowns), total_len, len(total_unknowns)/total_len))
        return res

    def texts_to_sequences_generator(self, texts):
        '''Transforms each text in texts in a sequence of integers.
        Only top "nb_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        Unknown words use an UNKNOWN index.
        Yields individual sequences.
        # Arguments:
            texts: list of strings.
        '''
        nb_words = self.nb_words
        
        for seq in texts:
            unknowns = []
            sent_len = len(seq)
            vect = []
            if self.padding:
                vect.append(self.word_index.get(START_OF_SENTENCE))
            for w in seq:
                w = w.strip()
                lower = True
                alnum = True
                if not w.islower():
                    lower = False
                if not w.isalnum():
                    alnum = False
                if self.lower:
                    w = w.lower()
                w = re.sub('\d', '0', w)
                i = self.word_index.get(w)
                if i is not None:
                    vect.append(i)
                else:
                    unknowns.append(w)
                    if self.nb_unknowns == 1:
                        vect.append(self.word_index.get(UNKNOWN))
                    else:
                        # self.nb_knowns = 3
                        if not alnum:
                            vect.append(self.word_index.get(UNKNOWN_NON_ALNUM))
                        elif lower:
                            vect.append(self.word_index.get(UNKNOWN_LOWERCASE_ALNUM))
                        else:
                            vect.append(self.word_index.get(UNKNOWN_UPPERCASE_ALNUM))
            if self.padding:
                vect.append(self.word_index.get(END_OF_SENTENCE))
            yield vect, unknowns, sent_len

