import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
import re
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from torchtext.vocab import GloVe
from torch.utils.data import Dataset
from nltk import word_tokenize
from nltk.corpus import stopwords, webtext
import nltk

from textfn import *

def EDA(df):
    print("-----------------------[Information]-------------------------") 
    print(df.info()) 
    print("-----------------------[Top of Data]-------------------------")
    print(df.head(3))
    print("-----------------------[NULL Elements]-------------------------")
    print(df.isnull().sum())
    print("-----------------------[NaN Elements]-------------------------")
    print(df.isna().sum())
    print("--------------------------------------------------------------")


    df.isnull().sum().plot.bar()
    plt.show()
    df['t_mean'] = df.groupby('keyword')['target'].transform('mean')
    fig = plt.figure(figsize=(8, 72), dpi=100)

    sns.countplot(y=df.sort_values(by='t_mean', ascending=False)['keyword'],
                hue=df.sort_values(by='t_mean', ascending=False)['target'])

    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=12)
    plt.legend(loc=1)
    plt.title('Target Distribution in Keywords')
    plt.show()

    df.drop(columns=['t_mean'], inplace=True)

    #Still need things:
    
    # word_count number of words in text
    # unique_word_count number of unique words in text
    # stop_word_count number of stop words in text
    # url_count number of urls in text
    # mean_word_length average character count in words
    # char_count number of characters in text
    # punctuation_count number of punctuations in text
    # hashtag_count number of hashtags (#) in text
    # mention_count number of mentions (@) in text

def cleaningProcessing(df):
    # Delete hashtags and @
    # df['text'] = df['text'].apply(r_hashtagsAt)
    # URLs
    df['text'] = df['text'].apply(r_url)
    # Lowercasing
    df['text'] = df['text'].apply(r_upper)
    # Punctuation & special Chars
    df['text'] = df['text'].apply(entity_ref)
    df['text'] = df['text'].apply(r_punctuation)
    # Expand contractions
    df['text'] = df['text'].apply(expand_contractions)
    # Numbers
    df['text'] = df['text'].apply(r_number)
    # Stopword cleaning
    df['text'] = df['text'].apply(word_tokenize)
    df['text'] = df['text'].apply(r_stopwords)
    # Stemming and/or Lemmatization
    # --- Lemmatization
    df['text'] = df['text'].apply(lemmatization)
    # --- Stemming
    # df['text'] = df['text'].apply(stemming)

    # ----- 
    # Dropping the location column - 
    # Want to keep it simple for now + it might now make big difference on accuracy
    # ----- 
    df = df.drop('location', axis=1)

    # Tranforming the keyword column
    df['keyword'] =  df['keyword'].apply(lambda x: 0 if type(x) == float else x)
    df['keyword'] =  df['keyword'].apply(lambda x: 0 if x == 0 else word_tokenize(x))
    return df 

def vectorization(df):
    # CountVector
    vectorizer = CountVectorizer()
    test = vectorizer.fit_transform(df['text'])
    # Tfid
    print(test)
    # GloVe
    # word2Vec
    return 0


if __name__ == '__main__':

    # Note for later:
    # Need to check if by processing the hashtags differently, 
    # this should give some better results 
    # (as for ex #earthquake might be good to keep)
    df = pd.read_csv("dataset/train.csv")
    df = cleaningProcessing(df)
    df.to_csv("dataset/train_processed.csv")




