import torch
import numpy as np
   
# --------------------------- MODELS -----------------------------

# --------------------------- DATASETS -----------------------------
    
from torch.utils.data import Dataset

class TranformerGloveDataset(Dataset):
    def __init__(self, df, max_seq_length, word_to_index, train=False):
        self.max_seq_length = max_seq_length
        self.train = train
        self.wti = word_to_index
        self.X = df['text']
        if train:
            self.Y = torch.from_numpy(np.array(df['target'])).float()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X = []
        if len(self.X[index]) >= self.max_seq_length:
            X = self.X[index][:self.max_seq_length]
        else:
            X = self.X[index] + ["<pad>"] * int(self.max_seq_length-len(self.X[index]))
        X = [self.wti[word] if word in self.wti else self.wti["<unk>"] for word in X]
        res = X
        if self.train:
            res = torch.tensor(X), self.Y[index]
        return res
