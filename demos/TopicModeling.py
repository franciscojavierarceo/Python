# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:08:01 2015

@author: franciscojavierarceo
"""

import os
import pandas as pd
from gensim import corpora, models, similarities
from collections import defaultdict
os.chdir('/Users/franciscojavierarceo/')
df1 = pd.read_csv('NYTimesdata.csv')
df1.head()
frequency = defaultdict(int)

documents = df1['Title'].values
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]
texts
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
            for text in texts]

from pprint import pprint   # pretty-printer
pprint(texts)

dictionary = corpora.Dictionary(texts)
print(dictionary)
print(dictionary.token2id)
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
new_vec
print(new_vec)
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)
print(corpus)
 
 
class MyCorpus(object):
   def __iter__(self):
       for line in open('NYTimesdata.csv'):
# assume there's one document per line, tokens separated by whitespace
           yield dictionary.doc2bow(line.lower().split())
           

corpus_memory_friendly = MyCorpus() # doesn't load the corpus into memory!
print(corpus_memory_friendly)

for vector in corpus_memory_friendly: # load one vector into memory at a time
    print(vector)

from gensim import corpora
corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)
corpus = gensim.matutils.Sparse2Corpus(scipy_sparse_matrix)
scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)

