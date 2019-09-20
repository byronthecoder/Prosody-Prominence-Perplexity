import parselmouth
import os
import glob
import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
import sys
from collections import Counter

import shutil  # copy files
from os import path

from scipy.stats.stats import pearsonr
# import statsmodels.graphics.api as smg

import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import *

import re


# create the pandas dataframe of the prosodic features with some words dropped

def concat_features(pf_path, stemming=True, stopword=True, contat=False):
    pfs_df = []
    for p in pf_path:
        pf_df = pd.read_csv(p, sep="\s+", index_col=0, engine='python')

        words = pf_df.index


        special_characters = pf_df.index.str.extractall(r'(?P<square>.*\[.*)')

        pf_df.drop(special_characters.square, inplace=True)

        pf_df.replace('--undefined--', np.nan, inplace=True)

        pf_df = pf_df.astype("float64")

        if stemming==True:
            st = LancasterStemmer()
            words = [st.stem(word) for word in words]
            pf_df.index = words



        pfs_df.append(pf_df)

    if contat == True:
        p_features = pd.concat(pfs_df)
    else:
        p_features = pfs_df
    return p_features

def get_all_sents(trans_paths, stemming=True):
    all_sents = []
    for txt in (trans_paths):
        with open(txt) as f:
            trans = f.readlines()

        sent_list = []
        for sent in trans:
            sent = re.findall(r'([A-Za-z\'\-\_\[\]]+)\s', sent)

            unwanted_words = {'[silence]', '[noise]', '[laughter]'}

            sent = [ele for ele in sent if ele not in unwanted_words]

            # insert sentence beginning
            #         sent.insert(0, '<s>')

            # stemming
            if stemming == True:
                st = LancasterStemmer()
                sent = [st.stem(word) for word in sent]
            else:
                break

            #         if stemmedWords != ['<s>']:
            #             sent_list.append(stemmedWords)

            sent_list.append(sent)

        all_sents.append(sent_list)
    return all_sents


# create the corpus

def get_corpus(trans_paths):
    all_sents = get_all_sents(trans_paths)

    # Flat the corpus

    corpus_flat = []
    for trans in all_sents:
        for sent in trans:
            for word in sent:
                corpus_flat.append(word)
    return corpus_flat


def get_ngram(corpus, n):
    if n == 1:
        unigram = Counter(corpus)
        return unigram
    elif n == 2:
        bigrams = Counter(list(ngrams(corpus, n)))
        return bigrams
    elif n == 3:
        trigrams = Counter(list(ngrams(corpus, n)))
        return trigrams
    elif n == 4:
        fourgrams = Counter(list(ngrams(corpus, n)))
        return fourgrams
    elif n == 5:
        fivegrams = Counter(list(ngrams(corpus, n)))
        return fivegrams
    else:
        print("Error: param n should be in range [1, 5]")

def uni_prob(p_features, corpus):
    p_features = p_features
    corpus = corpus
    unigram = get_ngram(corpus, 1)
    words = list(p_features.index)
    st = LancasterStemmer()
    words = [st.stem(word) for word in words]

    prob = []
    #     for token in words:
    #         prob1.append(unigram[token]/len(corpus))
    for token in words:
        if unigram[token] != 0:
            prob.append(log(unigram[token] / len(corpus)))

        else:
            prob.append(np.nan)

    prob_df = pd.DataFrame(prob, columns=['prob'])
    prob_df.index = words
    return prob_df


def bi_prob(p_features, corpus):
    prob = []
    p_features = p_features
    corpus = corpus

    #     unigram = uni_prob(p_features, corpus)
    unigram = get_ngram(corpus, 1)
    bigram = get_ngram(corpus, 2)
    words = list(p_features.index)
    st = LancasterStemmer()
    words = [st.stem(word) for word in words]
    bigram_words = list(ngrams(words, 2))

    #     use uni_prob for some missing value
    #     prob.append(unigram.prob.iloc[0])

    prob.append(np.nan)

    for token in bigram_words:
        bi_num = bigram[token]
        uni_num = unigram[token[0]]

        if bi_num != 0:
            bi_prob = log(bi_num / uni_num)
            prob.append(bi_prob)
        else:
            prob.append(np.nan)

    prob_df = pd.DataFrame(prob, columns=['prob'])
    prob_df.index = words
    return prob_df


# if __name__ == '__main__':

    pf_path = sorted(glob.glob('../swb1_LDC97S62/swb1_d1/3553B-3731B/*.means'))
    p_features = concat_features(pf_path)

    trans_paths = sorted(glob.glob('../switchboard_word_alignments/*/*/*/*trans.text'))
    corpus = get_corpus(trans_paths)

    prob_df_uni = uni_prob(p_features, corpus)

    for feature in p_features.columns[0:11]:
        print(feature)
        print(p_features[feature].corr(prob_df_uni.prob, method='spearman'))
        print(p_features[feature].corr(prob_df_uni.prob, method='pearson'))

    prob_df_bi = bi_prob(p_features, corpus)

    for feature in p_features.columns[0:11]:
        print(feature)
        print(p_features[feature].corr(prob_df_bi.prob, method='spearman'))
        print(p_features[feature].corr(prob_df_bi.prob, method='pearson'))

        import pickle

        with open('./features.pkl', 'wb') as f:
            pickle.dump(pf, f)

with open('../data/corpus_no_special_chars.txt', 'r') as f:
    corpus = f.readlines()
