import pandas as pd

# coding: utf-8

# In[1]:

import gensim

import numpy as np

from nltk import word_tokenize


model = gensim.models.KeyedVectors.load_word2vec_format('word2vec/vectors_small.txt', binary=False)

df = pd.read_csv('pre_processed_data.csv')
orig_df = df
df = list(df['0'])


def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.vocab]
    return np.mean(word2vec_model[doc], axis=0)

def preprocess(text):
    doc = word_tokenize(text)
    return doc

corpus = [preprocess(text) for text in df]

x =[]
for doc in df: #look up each doc in model
    x.append(document_vector(model, doc))

x = np.asarray(x)
x = pd.DataFrame(x)

df = np.asarray(df)
df = pd.DataFrame(df)

final_data = pd.concat([orig_df, x], axis=1)

final_data.to_csv('w2v.csv')