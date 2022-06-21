# %%
from ast import literal_eval
# from gain import gain
import pandas as pd
import os
from IPython.display import display
# import numpy as np
# from gensim.models import word2vec
from tqdm import tqdm
# %%
CSV_PATH = '..\..\data\df_jieba.csv'
W2V_LTP_PATH = '..\..\data\word2vec_jieba.model'
GAIN_PATH = '..\..\..\data\info_gain.csv'
# %%
# df = pd.read_csv(CSV_PATH)
# df = df.sample(frac=1).reset_index(drop=True)
gain_df = pd.read_csv(GAIN_PATH)
display(gain_df)
# %%
max = gain_df['information_gain'].max()
min = gain_df['information_gain'].min()
print(max, min)
# %%


def min_max(x):
    return (x - min)/(max-min)


# %%
gain_df['weight'] = gain_df['information_gain'].apply(lambda x: min_max(x))
# %%
display(gain_df)
# %%
