# %%
from ast import literal_eval
# from gain import gain
import pandas as pd
import os
from IPython.display import display
import numpy as np
from gensim.models import word2vec
from tqdm import tqdm
# %%
CSV_PATH = '..\..\data\df_jieba.csv'
W2V_LTP_PATH = '..\..\data\word2vec_jieba.model'
GAIN_PATH = '..\..\data\info_gain.csv'
# %%
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1).reset_index(drop=True)
gain_df = pd.read_csv(GAIN_PATH)
# %%
ask_df = df.loc[:, ['cat_id', 'ask_clean_w2v']]
# %%
w2v_model = word2vec.Word2Vec.load('word2vec.model')
# %%
for word in w2v_model.wv.index_to_key:
    for item in w2v_model.wv.most_similar(word):
        print(item)
# %%
display(df.head())
# %%


def weight(doc):
    weight_lst = []
    for word in doc:
        pass
# %%
