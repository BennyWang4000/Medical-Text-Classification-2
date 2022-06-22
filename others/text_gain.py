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
CSV_PATH = 'D:\CodeRepositories\py_project\data_mining\data\df_jieba_none.csv'
W2V_MODEL_PATH = 'D:\CodeRepositories\py_project\data_mining\data\word2vec_jieba_none.model'
# %%
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1).reset_index(drop=True)
# %%
display(df.head())

# %%
# ask_lst = [literal_eval(lst) for lst in ask_lst]
# %%
w2v_model = word2vec.Word2Vec.load(W2V_MODEL_PATH)

w2v_key_lst = w2v_model.wv.index_to_key
# %% ===============================================
# * remove that not in w2v model


lines_bar = tqdm(total=len(df.index), position=0, leave=True)


def remove_not_exist(x, is_tqdm=False):
    # words = [literal_eval(lst) for lst in x]
    words = literal_eval(x)
    return_lst = []
    if is_tqdm:
        lines_bar.update()
    for word in words:
        if word in w2v_key_lst:
            return_lst.append(word)

    return return_lst


df['ask_clean_w2v'] = df['ask_clean'].apply(
    lambda x: remove_not_exist(x, True))
display(df.head())
# %%

df.to_csv(os.path.join('..\..\data\df_jieba_none_remove.csv'))
# %% ===============================================
# %%
ask_df = df.loc[:, ['cat_id', 'ask_clean_w2v']]
# %%
display(ask_df.head())
# %%
vocab = set()
doc_vocab = []
number_of_terms = 0
number_of_docs = 0
class_dictionary = {}
cls_index = 0
doc_clss_index = []
count_of_that_class = []
class_name = []

# with open('data.txt', 'r', encoding="utf8") as infile:
# for line in infile:

for index, row in tqdm(df.iterrows(), total=len(df.index)):
    number_of_docs += 1
    cls = row['cat_id']
    words = literal_eval(row['ask_clean'])
    # assigning class index for each document
    if (class_dictionary.get(cls)) == None:
        class_dictionary[cls] = cls_index
        tmp = cls_index
        cls_index += 1
        count_of_that_class.append(1)
        class_name.append(cls)

    else:
        tmp = class_dictionary[cls]
        count_of_that_class[tmp] += 1
    doc_clss_index.append(tmp)

    w2v_key_lst = w2v_model.wv.index_to_key
    embeds = set()
    for word in words:
        if word in w2v_key_lst:
            # embeds.add(w2v_model.wv.key_to_index[word])
            embeds.add(word)
    tmp_set = set()
    # number_of_terms += len(tokens)
    for word in embeds:
        vocab.add(word)
        tmp_set.add(word)
    doc_vocab.append(tmp_set)
# %%
print(len(vocab))
# %%
vocab_size = len(vocab)
number_of_classes = cls_index
count_of_that_class = np.asarray(count_of_that_class)
probability_of_classess = count_of_that_class / number_of_docs
print(vocab_size, '\n', number_of_classes, '\n',
      count_of_that_class, '\n', probability_of_classess)
# %%
word_occurance_frequency = np.zeros(vocab_size, dtype=int)
word_occurance_frequency_vs_class = np.zeros(
    (vocab_size, number_of_classes), dtype=int)
word_index = {}
counter = -1
vocab_list = []
for word in vocab:
    counter += 1
    word_index[word] = counter
    vocab_list.append(word)
vocab_list = np.asarray(vocab_list)
for i in range(0, number_of_docs):
    for word in doc_vocab[i]:
        index = word_index[word]
        word_occurance_frequency[index] += 1
        word_occurance_frequency_vs_class[index][doc_clss_index[i]] += 1
# %% cal prob
p_w = word_occurance_frequency/number_of_docs
p_w_not = 1 - p_w
p_c = probability_of_classess

p_class_condition_on_w = np.zeros((number_of_classes, vocab_size), dtype=float)
tmp = word_occurance_frequency_vs_class.T
for i in range(0, number_of_classes):
    p_class_condition_on_w[i] = tmp[i]/word_occurance_frequency


p_class_condition_on_not_w = np.zeros(
    (number_of_classes, vocab_size), dtype=float)
for i in range(0, number_of_classes):
    p_class_condition_on_not_w[i] = (
        count_of_that_class[i]-tmp[i])/(number_of_docs-word_occurance_frequency)


# %%
word_ig_information = []
e_0 = 0.0
for c_index in range(0, number_of_classes):
    e_0 += p_c[c_index]*np.log2(p_c[c_index])
e_0 = -e_0
for w_index in range(0, vocab_size):
    e_1 = 0.0
    for c_index in range(0, number_of_classes):
        tmp1 = p_class_condition_on_w[c_index][w_index]
        if tmp1 != 0:
            e_1 += p_w[w_index]*tmp1*np.log2(tmp1)
        tmp2 = p_class_condition_on_not_w[c_index][w_index]
        if tmp2 != 0:
            e_1 += (1-p_w[w_index])*(tmp2*np.log2(tmp2))
    e_1 = -e_1

    information_gain = e_0 - e_1

    word_ig_information.append([information_gain, vocab_list[w_index]])

word_ig_information = sorted(
    word_ig_information, key=lambda x: x[0], reverse=True)

# %%
preview = pd.DataFrame(word_ig_information)
preview.columns = ['information_gain', 'word']

# %%
display(preview.sort_values(['information_gain'], ascending=False))
# %%
preview = preview.sort_values(['information_gain'], ascending=False)
# %%
preview.to_csv('../../data/info_gain_none.csv')
# %%
