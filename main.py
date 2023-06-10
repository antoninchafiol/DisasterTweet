import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt

from torch.utils.data import Dataset




if __name__ == '__main__':

    df = pd.read_csv("dataset/train.csv")
    print("-----------------------[Information]-------------------------") 
    print(df.info()) 
    print("-----------------------[Top of Data]-------------------------")
    print(df.head(3))
    print("-----------------------[NULL Elements]-------------------------")
    print(df.isnull().sum())
    print("-----------------------[NaN Elements]-------------------------")
    print(df.isna().sum())
    print("--------------------------------------------------------------")