from torch.utils.data import Dataset
from torch import load
import os
import pandas as pd
import glob

import torch
import numpy as np

class TextDataset(Dataset):

    def __init__(self, data_root):
        self.data_len = 0
        self.df = pd.DataFrame()

        for file_path in os.listdir(os.path.join(data_root, 'clean_data')):
            self.df = pd.concat(
                [self.df, pd.read_csv(file_path)], ignore_index=True)

        

    def __getitem__(self, index):

        
        return dep, word_lst

    def __len__(self):
        return self.data_len

class SparseFitDataset(Dataset):
    def __init__(self, features, targets):
        '''Dataset for train. Return features and labels.'''
        
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        cur_features = torch.from_numpy(self.features[idx].toarray()[0]).float()
        cur_label = torch.from_numpy(np.asarray(self.targets[idx])).long()
        return cur_features, cur_label
    