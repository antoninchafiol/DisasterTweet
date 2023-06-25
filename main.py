import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import F1Score
import pandas as pd
import numpy as np
import math
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from scipy.sparse import hstack, csr_matrix
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
    print(vectorizer.vocabulary_)
    # Tfid
    # GloVe
    # word2Vec
    return vectorized_textc, vectorized_keywordc

def getVocab(df=pd.read_csv("dataset/merged_data.csv")):
    vocab = CountVectorizer()
    vocab.fit_transform(df['text'])
    return vocab

class Cdataset():
    def __init__(self, df, vocab, train=False):
        self.train = train
        self.vocab = vocab
        self.seq_length = []
        self.text = df['text'] # Keep in mind that 
        for i in range(0, len(df['text'])):
            reformat = re.sub(r'\'|\[|\]|\s', '', df['text'][i]).split(',')
            # self.text.append(reformat)
            self.seq_length.append(len(reformat))
        vec = vocab.transform(self.text)
        self.X = torch.from_numpy(vec.todense()).float()
        if train==True:
            self.Y = torch.from_numpy(np.array(df['target'])).float()
    def __len__(self):
        return self.X.size()[0]
    def __getitem__(self, index):
        res = self.X[index], self.text[index] # Keep in mind that text is a string, not a list 
        if self.train==True:
            res = self.X[index], self.Y[index], self.text[index]
        return res
    def __shape__(self):
        return self.X.size()

def train(model, model_params, params, loader, mode='train', verbose=False):
    grad = True
    if mode=='train':
        model.train()
    elif mode=='eval':
        model.eval()
        grad = False

    g_acc = []
    g_loss = []
    for e in range(params['epochs']):
        acc = 0
        loss = 0
        for X, Y, _ in loader:
            model_params['optimizer'].zero_grad()
            X = X.to(params['device'])
            Y = Y.to(params['device'])
            X = torch.reshape(X, (X.size(0), 1, X.size(1)))
            with torch.set_grad_enabled(grad):
                y_hat = model(X)
                loss = model_params['loss_fn'](y_hat, Y)
                if mode=='train':
                    loss.backward()
                    model_params['optimizer'].step()
                acc += model_params['metric'](y_hat, Y)
                loss += loss.item()           
        e_acc = acc.detach().numpy()/len(loader)
        e_loss = loss.detach().numpy()/len(loader)
        g_acc.append(e_acc)
        g_loss.append(e_loss)
        if verbose:
            print('[{} Epoch {}] - Loss: {} - Acc: {}'.format(
                mode, e, np.round(e_loss, decimals=2), np.round(e_acc, decimals=2)))

    return model, acc, loss


if __name__ == '__main__':

    # Note for later:\w
    # Need to check if by processing the hashtags differently, 
    # this should give some better results 
    # (as for ex #earthquake might be good to keep)


    train_raw_text_df = pd.read_csv("dataset/train_processed.csv")
    test_raw_text_df = pd.read_csv("dataset/test_processed.csv")
    vocab = getVocab()
    params = {
        'epochs': 5,
        'batch_size': 256, 
        'input_dim':len(vocab.vocabulary_),
        'hidden_dim':50,
        'output_dim':1,
        'n_layers': 5,
        'dropout': 0.25,
        'bidirectional': False,
        'split_seed': 42,
        'train_dev_split': 0.65,
        'optim_lr': 0.001, 
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data = Cdataset(train_raw_text_df, vocab, train=True)
    test_data = Cdataset(test_raw_text_df, vocab, train=False)
    train_data, dev_data = torch.utils.data.random_split(train_data, [params['train_dev_split'], 1-params['train_dev_split']])

    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=False)
    
    model = SimpleLSTM(params['input_dim'], params['hidden_dim'], params['output_dim'], 
                       params['n_layers'], params['bidirectional'], params['dropout'], params['device']).to(params['device'])
    
    model_params = {
        'optimizer': torch.optim.RMSprop(model.parameters(), lr=params['optim_lr']), 
        'loss_fn'  : torch.nn.CrossEntropyLoss(),
        'metric'   : F1Score(task='binary').to(params['device'])
    }

    # Train
    model, train_acc, train_loss = train(model, model_params, params, train_loader, mode='train', verbose=True)
    # Eval 
    model, train_acc, train_loss = train(model, model_params, params, dev_loader, mode='eval', verbose=True)
    # Display some graphs
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot([i for i in range(n_epoch)], t_acc, color='green', label='Train')
    # ax1.plot([i for i in range(n_epoch)], d_acc, color='red', label='Dev')
    # ax1.set_ylabel('Accuracy')
    # ax2.plot([i for i in range(n_epoch)], t_loss, color='green', label='Train')
    # ax2.plot([i for i in range(n_epoch)], d_loss, color='red', label='Dev')
    # ax2.set_ylabel('Loss')
    # plt.show()
    # # Test data
    # test_model = SimpleLSTM(input_len, hidden_size, output_size, num_layers, bidirectional, dropout, device).to(device)
    # test_model.load_state_dict(model.state_dict())
    # for e in range(n_epoch):
    #     for X, _ in test_loader:
    #         model.eval()
    #         with torch.no_grad():
    #             X = X.to(device)
    #             output = model(X)




