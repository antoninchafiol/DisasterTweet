import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import F1Score
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from torchtext.vocab import GloVe
from torch.utils.data import Dataset
from nltk import word_tokenize
from nltk.corpus import stopwords, webtext
import nltk

from textfn import *
from classes import *

def EDA(df): # Really Quick EDA
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
    df['text'] = df['text'].apply(r_hashtagsAt)
    # URLs
    df['text'] = df['text'].apply(r_url)
    # Punctuation & special Chars
    df['text'] = df['text'].apply(r_specialChar)
    df['text'] = df['text'].apply(entity_ref)
    df['text'] = df['text'].apply(r_punctuation)
    # Lowercasing
    df['text'] = df['text'].apply(r_upper)
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
    df['keyword'] = df['keyword'].apply(lambda x: 0 if type(x) == float else x)
    df['keyword'] = df['keyword'].apply(lambda x: 0 if x == 0 else re.sub(r'[0-9]', "",x))
    df['keyword'] = df['keyword'].apply(lambda x: 0 if x == 0 else r_punctuation(x))
    df['keyword'] = df['keyword'].apply(lambda x: 0 if x == 0 else word_tokenize(x))
    df['keyword'] = df['keyword'].apply(lambda x: 0 if x == 0 else lemmatization(x))

    return df 

def vectorization(df):
    # CountVector
    vectorizer = CountVectorizer()
    vectorized_textc = vectorizer.fit_transform(df['text'])
    vectorized_keywordc = vectorizer.fit_transform(df['keyword'])
    # Tfid
    # GloVe
    # word2Vec
    return vectorized_textc, vectorized_keywordc

class Cdataset():
    def __init__(self, df):
        X_train, _ = vectorization(df)
        self.X = torch.from_numpy(X_train.todense()).float()
        self.Y = torch.from_numpy(np.array(df['target'])).float()
    def __len__(self):
        return self.Y.size()[0]
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

if __name__ == '__main__':

    # Note for later:\
    # Need to check if by processing the hashtags differently, 
    # this should give some better results 
    # (as for ex #earthquake might be good to keep)

    df = pd.read_csv("dataset/test.csv")
    df = cleaningProcessing(df)
    df.to_csv('dataset/test_processed.csv')
    # batch_size=128
    # n_epoch = 100
    # input_len = 11501 # Taken from dict size 
    # hidden_size = 3
    # output_size = 1
    # lr = 0.01
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # df = pd.read_csv("dataset/train_processed.csv")
    # ds = Cdataset(df)
    # train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # model = SimpleNet(input_len, hidden_size, output_size)
    # model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # f1 = F1Score(task='binary').to(device)

    # lacc = []
    # lloss = []
    # for e in range(n_epoch):
    #     e_acc = 0
    #     e_loss = 0
    #     for X, Y in train_loader:
    #         X = X.to(device)
    #         Y = Y.to(device)
    #         optimizer.zero_grad()

    #         model.train()
    #         train_acc = 0
    #         with torch.set_grad_enabled(True):
    #             y_hat = model(X)
    #             loss = loss_fn(y_hat, Y)
    #             loss.backward()
    #             optimizer.step()
    #             train_acc = f1(y_hat, Y)

    #         model.eval()
    #         eval_acc = 0
    #         with torch.set_grad_enabled(False):
    #             y_hat = model(X)
    #             loss = loss_fn(y_hat, Y)
    #             e_acc  += loss.item() 
    #             e_loss += f1(y_hat, Y)
    #             eval_acc = f1(y_hat, Y)
    #     lacc.append(e_acc/train_loader.__len__())
    #     lloss.append(e_loss/train_loader.__len__())
    #     if e%10==0:
    #         print('After {} epoch training loss is {}, Train F1 is {} - Eval F1: {}'.format(e,loss.item(), train_acc, eval_acc))
    # # df.to_csv("dataset/train_processed.csv")

    # # Test data
    # test_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

