# %%
from ast import literal_eval
from gensim.models import word2vec
import pandas as pd
import os
from IPython.display import display
import numpy as np
# %%
CSV_PATH = 'D:\CodeRepositories\py_project\data_mining\data\df_jieba_none.csv'
df = pd.read_csv(CSV_PATH)
# %%
display(df.head())
# %%
ask_lst = df['ask_clean'].tolist()
# %%
print((ask_lst[0]))
# %%
ask_lst = [literal_eval(lst) for lst in ask_lst]
# %%
seed = 555
sg = 0
window_size = 5
vector_size = 100
min_count = 5
workers = 4
epochs = 5
batch_words = 10000
# %%
model = word2vec.Word2Vec(
    ask_lst,
    min_count=min_count,
    vector_size=vector_size,
    workers=workers,
    epochs=epochs,
    window=window_size,
    sg=sg,
    seed=seed,
    batch_words=batch_words
)
# %%
model.save(
    'D:\CodeRepositories\py_project\data_mining\data\word2vec_jieba_none.model')

# %%
