# %%
from torch.utils.data import Dataset
import pandas as pd
from gensim.models import word2vec
from ast import literal_eval

import torch
import numpy as np

import sys
if '../' not in sys.path:
    sys.path.append('../')
from config import Config
# %%


class TextDataset(Dataset):
    def __init__(self, cfg: Config):
        '''TextDataset: torch.utils.data.Dataset
        param:
            data_path: str
            w2v_model_path: str,
            weight_path: str,
        '''
        self.data_df = pd.read_csv(cfg.data_path)
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)
        self.data_df = self.data_df.loc[:, ['cat_id', 'ask_clean_w2v']]
        self.w2v_model = word2vec.Word2Vec.load(cfg.w2v_model_path)
        self.weight_df = pd.read_csv(cfg.weight_path)

        self.data_len = len(self.data_df.index)
        self.cfg = cfg

    def __getitem__(self, index):
        '''
        param:
            index

        return:
            cat_id: 1x1 torch.Tensor, category id
            words: nx100 torch.Tensor, 100 dimensions word embedding after weighing
            leng: int, length of document
        '''
        row = self.data_df.iloc[[index]]
        words = literal_eval(row['ask_clean_w2v'].tolist()[0])

        cat_id = np.zeros((2))

        cat_id[row['cat_id']] = 1

        weights = np.zeros((self.cfg.words_len, 1))
        words_vec = np.zeros((self.cfg.words_len, self.cfg.embed_dim))

        for index, word in enumerate(words[:40]):
            if word in self.w2v_model.wv.index_to_key:
                words_vec[index] = self.w2v_model.wv[word]
                weights[index] = self.weight_df.loc[self.weight_df['word']
                                                    == word, 'weight']

        weights = weights.reshape(-1, 1)

        return {'cat_id': torch.from_numpy(cat_id).float().to(self.cfg.device),
                'words': torch.from_numpy(weights * words_vec).float().to(self.cfg.device),
                'leng': weights.shape[0]}

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
        cur_features = torch.from_numpy(
            self.features[idx].toarray()[0]).float()
        cur_label = torch.from_numpy(np.asarray(self.targets[idx])).long()
        return cur_features, cur_label


# # %%
# %%

# lst = [1, 2, 3, 4]

# print(lst[:2])
# if True:
#     import numpy as np

# v=np.zeros((40, 1))
# for i in range(0, 40):
#     v[i]=i
# w=np.zeros((40, 100))
# for i in range(0, 40):
#     w[i]=np.arange(100)
# print(w)
# print(v.shape, w.shape)
# # print(v)
# # print(w)
# print(w*v)
# %%
# n = np.empty(0)

# n = np.append(n, np.arange(5))
# n = np.vstack((n, np.arange(5)))
# n = np.vstack((n, np.arange(5)))

# n = np.concatenate((n, np.arange(5)), axis=0)
# n = np.concatenate((n, np.arange(5)), axis=0)
# n = np.concatenate((n, np.arange(5)), axis=0)

# print(n)
# print(n.shape)
# %%
# v = np.arange(9.0).reshape((3, 3))
# print(v)
# w = np.arange(3.0)
# print(w)
# print(w * v)

# # %%
# w = np.empty(0)
# for i in range(10):
#     w = np.append(w, i)
# print(w)
# # %%
