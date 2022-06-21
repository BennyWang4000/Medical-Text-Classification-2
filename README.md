# Medical-Text-Classification-2
Expected to add selection of feature words 
## TODO
- feature words selection
- words weight
- stemming
- feature words weight
## Problem
- too much word
    - feature selection
        - chi2
        - filter, wrapper, embedded, hybrid 
        - infomation gain
- xgboost is inappreciate: underfitting
    - SVM
    - CNN
- result is not a result
    - use prob
    - CNN
## Pipeline
- word segment
- remove stopwords
- word2vec
- word weight
    - information gain
    - weight

![](https://cdn.discordapp.com/attachments/747728438814703616/988419415114666054/unknown.png)

|||
---|---
w(t, d)|word t weight in document d
tf(t, d)|frequency of word t appears in document d
N|total number of documents
n_t|number of documents in which the word t appears
phi(d)|class that document d belongs to
P(phi(d) \| t)|the probability that the word t appears in the class phi(d)
1 + P(phi(d) \| t)|to enhance the ability of the vector to distinguish text class


![feature-selection-medium](https://towardsdatascience.com/feature-selection-on-text-classification-1b86879f548e)