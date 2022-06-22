# %%
import matplotlib.pyplot as plt
from ast import literal_eval
# from gain import gain
import pandas as pd
import os
from IPython.display import display
import numpy as np
from gensim.models import word2vec
from tqdm import tqdm
# %%
CSV_PATH = 'D:\CodeRepositories\py_project\data_mining\data\df_jieba_none_remove.csv'
W2V_MODEL_PATH = 'D:\CodeRepositories\py_project\data_mining\data\word2vec_jieba_none.model'
GAIN_PATH = 'D:\CodeRepositories\py_project\data_mining\data\info_gain_none.csv'
# %%
df = pd.read_csv(CSV_PATH)
display(df.head())
# df = df.sample(frac=1).reset_index(drop=True)
# %%
'''
NOTE max= 904 ?? everage= 18.13
'''
# max = 0
# total = 0
leng_lst = []
for i, row in tqdm(df.iterrows(), total=len(df.index)):
    words = literal_eval(row['ask_clean_w2v'])
    leng_lst.append(len(words))
    # total += leng
    # if max < leng:
    #     max = leng
# %%
plt.figure(figsize=(15, 5))
plt.plot(leng_lst[49000:50000])
plt.show
# %%
gain_df = pd.read_csv(GAIN_PATH)
# %%
gain_df['weight'] = 0
display(gain_df.head())
# # %%
# ask_df = df.loc[:, ['cat_id', 'ask_clean_w2v']]
# %%
w2v_model = word2vec.Word2Vec.load(W2V_MODEL_PATH)
# %%
for word in w2v_model.wv.index_to_key:
    for item in w2v_model.wv.most_similar(word):
        print(item)
# %%
print(w2v_model.wv.most_similar('痔疮'))
# %%
display(gain_df.head())
for index, row in tqdm(gain_df.iterrows(), total=len(gain_df.index)):
    info_gain = row['information_gain']
    word = row['word']

    if gain_df.at[index, 'weight'] == 0:
        # print(word, info_gain)
        gain_df.at[index, 'weight'] = info_gain

    for i in w2v_model.wv.most_similar(word):
        simi_word, similarity = i
        try:
            simi_index = gain_df.loc[gain_df['word'] == simi_word].index[0]
        except:
            print('err', simi_word, similarity)
            continue

        if gain_df.at[simi_index, 'weight'] == 0:
            weight = gain_df.at[simi_index, 'information_gain']
            gain_df.at[simi_index, 'weight'] = weight + \
                abs(info_gain - weight) * similarity

        '''
        if weight is a zero:
            weight = itself+ (itself- wordinfo)* similarity
        
        
        '''
    # break
    # break

# %%
gain_df.to_csv(
    'D:\CodeRepositories\py_project\data_mining\data\info_gain_none_weight.csv')
# %%
# NOTE normalization
max = gain_df['weight_zero'].max()
min = gain_df['weight_zero'].min()
print(max, min)


def min_max(x):
    return (x - min)/(max-min)
# %% HACK padding zero


for index, row in tqdm(gain_df.iterrows(), total=len(gain_df.index)):
    weight = row['weight']
    info_gain = row['information_gain']
    word = row['word']

    if gain_df.at[index, 'weight'] == 0:
        gain_df.at[index, 'weight_zero'] = info_gain
    else:
        gain_df.at[index, 'weight_zero'] = weight

# %%
gain_df['weight_zero_normal'] = gain_df['weight_zero'].apply(
    lambda x: min_max(x))
# %%
display(gain_df.head(1000))
# %%
gain_df = gain_df.sort_values(['weight_zero_normal'], ascending=False)
# %%
gain_df.to_csv(
    'D:\CodeRepositories\py_project\data_mining\data\info_gain_none_weight_nor.csv')

# %%

weight = gain_df.loc[:, ['word', 'weight_zero_normal']]
weight.plot(title='word weight', xlabel='word',
            ylabel='weight', figsize=(10, 5))
plt.show()

# %%
