import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import re
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from torchtext.vocab import GloVe
from torch.utils.data import Dataset

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

def cleaning(text):
    # Delete hashtags and @
    r_hashtagsAt(text) 
    # Lowercasing
    r_upper(text)
    # Punctuation & special Chars
    r_punctuation(text)
    # Expand contractions
    expand_contractions(text)
    # URLs
    r_url(text)

    # Stopword cleaning
    # Lemmatization
    # 
    return 0 

def embedding(df):
    test = df['text']
    # CountVector
    vectorizer = CountVectorizer()
    vectorized = vectorizer.fit_transform(test)
    print(vectorized[0].toarray())
    # Tfid
    # GloVe
    # word2Vec
    return 0 

if __name__ == '__main__':
    df = pd.read_csv("dataset/train.csv")
    # print(df)
    # for tweet in df['text']:

    # embedding(df)
    print(r_upper("AAAAAAAAAAAAAA9Test 9@skdjfhjksdhfjkh .e;n/c'o#r]e u55468792314n aut[r$e%^&e*&s(t) avec #cewlinedionBBBBBBBBBBBBB "))
    # EDA(df)
