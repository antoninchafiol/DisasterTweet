import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import F1Score
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from torchtext.vocab import GloVe
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

def train(model, model_params, params, loader, mode='train', verbose=False):
    g_acc = {
        'train': [],
        'eval': []
    }
    g_loss = {
        'train': [],
        'eval': []
    }
    if mode!='both':
        if mode=='train':
            model.train()
        elif mode=='eval':
            model.eval()

        for e in range(params['epochs']):
            acc = 0
            loss = 0
            for X, Y, _ in loader[mode]:
                model_params['optimizer'].zero_grad()
                X = X.to(params['device'])
                Y = Y.to(params['device'])
                X = torch.reshape(X, (X.size(0), 1, X.size(1)))
                with torch.set_grad_enabled(mode=='train'):
                    y_hat = model(X)
                    loss = model_params['loss_fn'](y_hat, Y)
                    if mode=='train':
                        loss.backward()
                        model_params['optimizer'].step()
                    acc += model_params['metric'](y_hat, Y)
                    
                    loss += loss.item()           
            e_acc = acc.detach().numpy()/len(loader[mode])
            e_loss = loss.detach().numpy()/len(loader[mode])
            g_acc[mode].append(e_acc)
            g_loss[mode].append(e_loss)
            if verbose:
                print('[{} Epoch {}] - Loss: {} - Acc: {}'.format(
                    mode, e, np.round(e_loss, decimals=2), np.round(e_acc, decimals=2)))
    elif mode=='both':
        for e in range(params['epochs']):
            for loop_mode in 'train', 'eval':
                acc = 0
                loss = 0
                for X, Y, _ in loader[loop_mode]:
                    model_params['optimizer'].zero_grad()
                    X = X.to(params['device'])
                    Y = Y.to(params['device'])
                    with torch.set_grad_enabled(loop_mode=='train'):
                        y_hat = model(X)
                        # print(y_hat)
                        # print(Y)
                        loss = model_params['loss_fn'](y_hat, Y)
                        if loop_mode=='train':
                            loss.backward()
                            model_params['optimizer'].step()
                        acc += model_params['metric'](y_hat, Y)
                        # print(model_params['metric'](y_hat, Y))
                        loss += loss.item()           
                e_acc = acc.cpu().detach().numpy()/len(loader[loop_mode])
                # print(e_acc)
                e_loss = loss.cpu().detach().numpy()/len(loader[loop_mode])
                g_acc[loop_mode].append(e_acc)
                g_loss[loop_mode].append(e_loss)

            if verbose:
                print('Epoch {} - [Train] Loss: {} - Acc: {}, [Eval] Loss: {} - Acc: {}'.format(
                    e, 
                    np.round(g_loss['train'][-1], decimals=2), np.round(g_acc['train'][-1], decimals=2),
                    np.round(g_loss['eval'][-1], decimals=2),  np.round(g_acc['eval'][-1], decimals=2)))
            
    return model, g_acc, g_loss


if __name__ == '__main__':

    # Note for later:\w
    # Need to check if by processing the hashtags differently, 
    # this should give some better results 
    # (as for ex #earthquake might be good to keep)


    train_raw_text_df = pd.read_csv("dataset/train_processed.csv")
    test_raw_text_df = pd.read_csv("dataset/test_processed.csv")
    vocab = getVocab()
    params = {
        # Global
        'epochs': 10000,
        'batch_size': 128, 
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'split_seed': 42,
        'train_dev_split': 0.65,
        # Vocabulary 
        'seq_length': 35, 
        'embedding_dim': 50, 
        # Model
        'hidden_dim':10,
        'output_dim':1,
        'n_layers': 5,
        'dropout': 0.25,
        'bidirectional': False,
        # Optimizer
        'optim_lr': 0.01, 
    }



    embeddings = {}
    with open('glove_pretrained/glove.6B.{}d.txt'.format(params['embedding_dim']), "r", encoding="utf-8") as f:
        for line in f:
            # print(line)
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector

    # train_data = Cdataset(train_raw_text_df, vocab, train=True)
    # test_data  = Cdataset(test_raw_text_df, vocab,  train=False)
    # train_data, dev_data = torch.utils.data.random_split(train_data, [params['train_dev_split'], 1-params['train_dev_split']])
    # model = SimpleLSTM(params['vocab_size'], params['hidden_dim'], params['output_dim'], 
    #                    params['n_layers'], params['bidirectional'], params['dropout'], params['device']).to(params['device'])

    # # model = SimpleLSTMEmbedded(params['vocab_size'], params['embedding_dim'], params['hidden_dim'], params['output_dim'], 
    # #                    params['n_layers'], params['bidirectional'], params['dropout'], params['device']).to(params['device'])

    # loaders = {
    #     'train': DataLoader(train_data, batch_size=params['batch_size'], shuffle=True),
    #     'eval' : DataLoader(dev_data,   batch_size=params['batch_size'], shuffle=True),
    #     'test' : DataLoader(test_data,  batch_size=params['batch_size'], shuffle=False)
    # }

    # model_params = {
    #     'optimizer': torch.optim.Adam(model.parameters(), lr=params['optim_lr']), 
    #     'loss_fn'  : torch.nn.BCELoss(),
    #     'metric'   : F1Score(task='binary').to(params['device'])
    # }

    # # Train
    # model, train_acc, train_loss = train(model, model_params, params, loaders, mode='both', verbose=True)

    # Display some graphs
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot([i for i in range(params['epochs'])], acc['train'], color='green', label='Train')
    # ax1.plot([i for i in range(params['epochs'])], acc['eval'], color='red', label='Dev')
    # ax1.set_ylabel('Accuracy')
    # ax2.plot([i for i in range(params['epochs'])], loss['train'], color='green', label='Train')
    # ax2.plot([i for i in range(params['epochs'])], loss['eval'], color='red', label='Dev')
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




