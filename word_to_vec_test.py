# %%
from ast import literal_eval
from gensim.models import word2vec
import pandas as pd
import os
from IPython.display import display
import numpy as np
# %%
CSV_PATH = 'data\df_ltp_3.csv'
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
min_count = 1
workers = 4
epochs = 5
batch_words = 10000

train_data = word2vec.LineSentence('./all_e_ws.txt')
model = word2vec.Word2Vec(
    train_data,
    min_count=min_count,
    vector_size=vector_size,
    workers=workers,
    epochs=epochs,
    window=window_size,
    sg=sg,
    seed=seed,
    batch_words=batch_words
)

model.save('word2vec.model')
