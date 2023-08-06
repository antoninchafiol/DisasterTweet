import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import re
import numpy as np
   
# --------------------------- MODELS -----------------------------

class SimpleNet(torch.nn.Module):
    def __init__(self, input_len, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_len, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.output= torch.nn.Sigmoid()
    def forward(self, x):
        fc1 = self.fc1(x)
        fc2 = self.fc2(fc1)
        output = self.output(fc2)
        return output[:, -1]

class SimpleLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.bidirectional_val = 2 if bidirectional == True else 1 

        self.lstm1 = torch.nn.LSTM(input_dim, 
                                   hidden_dim, 
                                   num_layers = num_layers,
                                   bidirectional = bidirectional,
                                   batch_first = True
                                   )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hidden_dim * self.bidirectional_val, output_dim) 
        self.output= torch.nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
                                   batch_size,  self.hidden_dim).to(self.device))
        c_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
                                   batch_size,  self.hidden_dim).to(self.device))

        # Propagate input through LSTM
        _, (h_out, _) = self.lstm1(x, (h_0, c_0))

        # Concatenating bidirectionnal layers method --> Work but dsoesn't look better than 
        # common unidirectional one and not 100% sure if it's the rigfht way
        h_out = torch.cat((h_out[-1],h_out[-2]), dim=-1).squeeze(0)
        out = self.dropout(h_out)
        out = self.fc1(out)
        out = self.output(out)
        return out.squeeze(1) 

class MultiLSTM(torch.nn.Module):
    def __init__(self, input_dim, input_dim2, hidden_dim, hidden_dim2, output_dim, num_layers, bidirectional, dropout, device ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.num_layers = num_layers
        self.device = device
        self.bidirectional_val = 2 if bidirectional == True else 1 

        self.lstm1 = torch.nn.LSTM(input_dim, 
                                   hidden_dim, 
                                   num_layers = num_layers,
                                   bidirectional = bidirectional,
                                   batch_first = True
                                   )
        self.fc1 = torch.nn.Linear(hidden_dim * self.bidirectional_val, input_dim2) 
        self.dropout1 = torch.nn.Dropout(dropout)
        self.lstm2 = torch.nn.LSTM(input_dim2, 
                                   hidden_dim2, 
                                   num_layers = num_layers,
                                   bidirectional = bidirectional,
                                   batch_first = True
                                   )
        self.fc2 = torch.nn.Linear(hidden_dim2 * self.bidirectional_val, output_dim)
        self.output= torch.nn.Sigmoid()
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
                                   x.size(0),  self.hidden_dim).to(self.device))
        c_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
                                   x.size(0),  self.hidden_dim).to(self.device))

        # Propagate input through LSTM
        _, (h_out, _) = self.lstm1(x, (h_0, c_0))
        h_out = torch.cat((h_out[-1],h_out[-2]), dim=-1).squeeze(0)
        out = self.dropout1(h_out)
        out = self.fc1(out)

        h_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
                                   out.shape[0],  self.hidden_dim2).to(self.device))
        c_0 = Variable(torch.zeros(self.num_layers * self.bidirectional_val, 
                                   out.shape[0],  self.hidden_dim2).to(self.device))
        
        out = torch.reshape(out, (out.size(0), 1,  out.size(1)))
        _, (h_out, _) = self.lstm2(out, (h_0, c_0))
        h_out = torch.cat((h_out[-1],h_out[-2]), dim=-1).squeeze(0)
        out = self.fc2(out)
        out = self.output(out)
        out = out.squeeze(1)
        return out.squeeze(1)
    
class SimpleLSTMGloVe(torch.nn.Module):
    # This is just a copy of the SimpleLSTM class but using the torch.nn.Embedding layer
    def __init__(self, params, embed_weights):
        super().__init__()
        self.hidden_dim = params['hidden_dim']

        self.embed = torch.nn.Embedding(params['vocab_size'], params['embedding_dim'])
        self.embed.weight.data.copy_(embed_weights)
        self.embed.weight.requires_grad = False # Hard freezed, to keep glove weights
        self.lstm1 = torch.nn.LSTM(params['embedding_dim'], 
                                   params['hidden_dim'],
                                   batch_first = True
                                   )
        self.fc1 = torch.nn.Linear(params['hidden_dim'], params['output_dim']) 
        self.output= torch.nn.Sigmoid()

    def forward(self, x):  
        x = self.embed(x)
        out, (_, _) = self.lstm1(x)
        out = out[:, -1, :]
        out = self.fc(out).squeeze(1)
        out = self.output(out)
        return out
    
# --------------------------- DATASETS -----------------------------
    
class Cdataset():
    def __init__(self, df, vocab, train=False):
        self.train = train
        self.vocab = vocab
        self.seq_length = []
        self.text = df['text'] 
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
        res = self.X[index], self.text[index]
        if self.train==True:
            res = self.X[index], self.Y[index], self.text[index]
        return res
    def __shape__(self):
        return self.X.size()
    
class CdatasetGlove(Dataset):
    def __init__(self, df, max_seq_length, train=False):
        self.max_seq_length = max_seq_length
        self.train = train
        self.X = df['text']
        if train:
            self.Y = df['target'].tolist()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        res = self.X[index][:self.max_seq_length]
        if self.train:
            res = self.X[index][:self.max_seq_length], self.Y[index]
        return res
