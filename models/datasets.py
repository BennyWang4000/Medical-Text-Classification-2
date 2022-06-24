# %%
from torch.utils.data import Dataset
import pandas as pd
from gensim.models import word2vec
from ast import literal_eval

import torch
import numpy as np

# %%


class TextDataset(Dataset):
    def __init__(self, **cfg):
        '''TextDataset: torch.utils.data.Dataset
        param:
            data_path: str
            w2v_model_path: str,
            weight_path: str,
        '''
        self.data_df = pd.read_csv(cfg['data_path'])
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)
        self.data_df = self.data_df.loc[:, ['cat_id', 'ask_clean_w2v']]
        self.w2v_model = word2vec.Word2Vec.load(cfg['w2v_model_path'])
        self.weight_df = pd.read_csv(cfg['weight_path'])

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

        weights = np.zeros((self.cfg['words_len'], 1))
        words_vec = np.zeros((self.cfg['words_len'], self.cfg['embed_dim']))

        index = 0
        for word in words[:40]:
            if word in self.w2v_model.wv.index_to_key:
                words_vec[index] = self.w2v_model.wv[word]
                try:
                    weights[index] = self.weight_df.loc[self.weight_df['word']
                                                        == word, 'weight_zero_normal']
                except:
                    weights[index] = 0.00002
                index += 1

#         weights = weights.reshape(-1, 1)

        return {'cat_id': torch.from_numpy(cat_id).float().to(self.cfg['device']),
                'words': torch.from_numpy(weights * words_vec).float().to(self.cfg['device']),
                'leng': weights.shape[0]}

    def __len__(self):
        return self.data_len
