# %%
from tqdm import tqdm
# from harvesttext.resources import get_baidu_stopwords
import jieba
import pandas as pd
from IPython.display import display

stopwords_path = 'D:\CodeRepositories\py_project\data_mining\data\stopwords.txt'
csv_path = 'D:\CodeRepositories\py_project\data_mining\data\df_jieba_remove.csv'

# %%
df = pd.read_csv(csv_path, encoding='utf-8')

display(df.head())
# %%
none_df = df.loc[df['ask_clean'] == '[\'无\']']
test_df = df.loc[df['ask_clean'] == '[\'无\']'].head()

display(len(df.index))
display(len(none_df.index))
# display(test_df.head())
# %%

lines_bar = tqdm(total=len(df.index), position=0, leave=True)
# %%


def _remove_stop_words(words, stopwords_path):
    result = []
    stopwords = set(line.strip()
                    for line in open(stopwords_path, encoding='utf-8'))
    for word in words:
        if word not in stopwords:
            if word != ' ':
                result.append(word)
    return result


def word_segment(sentence, stopwords_path, is_tqdm=False):
    words = jieba.cut(sentence)
    words = _remove_stop_words(words, stopwords_path)
    if is_tqdm:
        lines_bar.update()
    return words


# %%
df.loc[df['ask'] == '无', 'ask_clean'] = df.loc[df['ask'] == '无', 'title'].apply(
    lambda x: word_segment(x, stopwords_path, True))
df.to_csv('D:\CodeRepositories\py_project\data_mining\data\df_jieba_none.csv')
# %%
