# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:34:56 2015

@author: franciscojavierarceo
"""
from sklearn.feature_extraction import DictVectorizer

measurements = [{'city': 'Dubai', 'temperature': 33.},
                {'city': 'London', 'temperature': 12.},
                {'city': 'San Fransisco', 'temperature': 18.}]
vec = DictVectorizer()

print measurements 
print vec.fit_transform(measurements).toarray()
print vec.get_feature_names()

pos_window = [
     {
             'word-2': 'the',
             'pos-2': 'DT',
             'word-1': 'cat',
             'pos-1': 'NN',
             'word+1': 'on',
             'pos+1': 'PP',
             },
             # in a real application one would extract many such dictionaries
             ]

vec = DictVectorizer()
pos_vectorized = vec.fit_transform(pos_window)
print vec.fit_transform(pos_window)
print pos_vectorized.toarray()
print vec.get_feature_names()

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
corpus = ['This is the first document.',
           'This is the second second document.',
           'And the third one.',
           'Is this the first document?',
           ]
X = vectorizer.fit_transform(corpus)
print X
X = vectorizer.fit_transform(corpus)
analyze("This is a text document to analyze.") == (['this', 'is', 'text', 'document', 'to', 'analyze'])

print X.toarray() 
vectorizer.vocabulary_.get('document')
vectorizer.transform(['Something completely new.']).toarray()
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
analyze('Bi-grams are cool!') == (['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])
X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
print X_2
feature_index = bigram_vectorizer.vocabulary_.get('is this')
print X_2[:, feature_index]     