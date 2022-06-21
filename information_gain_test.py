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
