# %%
from tqdm import tqdm
# from harvesttext.resources import get_baidu_stopwords
from ltp import LTP
import pandas as pd
from IPython.display import display

CSV_PATH = 'data\df_ltp_3.csv'
STOP_WORDS_PATH = 'data\stopwords.txt'

ltp = LTP()
# %%
df = pd.read_csv(CSV_PATH, encoding='utf-8')

display(df.head())
# %%
display(df.loc[df['ask'] == '无'].head())
# %%
display(df.loc[df['ask'] == '无']['title'].apply(
    lambda x: word_segment(x, STOP_WORDS_PATH, True)))
# %%

df.loc[df['ask'] == '无', 'ask_clean'] = df.loc[df['ask'] == '无']['title'].apply(
    lambda x: word_segment(x, STOP_WORDS_PATH))
display(df.head())
# %%
print()


# %%
def cat_id(x):
    if x == '內科':
        return 0
    else:
        return 1


df['cat_id'] = df['category'].apply(lambda x: cat_id(x))
# %%
df.drop('category', inplace=True, axis=1)
df.drop('department', inplace=True, axis=1)
# %%
display(df.head())

# %%
df = df[['cat_id', 'dep_id', 'cat_dep', 'title', 'ask', 'answer', 'ask_clean']]
# %%
lines_bar = tqdm(total=len(df.index), position=0, leave=True)
df['ask_clean'] = df['ask'].apply(
    lambda x: word_segment(x, STOP_WORDS_PATH, True))
# %%
display(df.head())
# %%


def _remove_stop_words(words, stopwords_path):
    result = []
    stopwords = set(line.strip()
                    for line in open(stopwords_path, encoding='utf-8'))
    for word in words:
        if word not in stopwords:
            result.append(word)
    return result


def word_segment(sentence, stopwords_path, is_tqdm=False):
    words = ltp.seg([sentence])[0][0]
    words = _remove_stop_words(words, stopwords_path)
    if is_tqdm:
        lines_bar.update()
    return words


# %%
df.to_csv('data\df_ltp_3.csv', index=False)
# %%
