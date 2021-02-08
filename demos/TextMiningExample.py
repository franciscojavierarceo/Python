# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 15:27:11 2015

@author: franciscojavierarceo
"""
import os
import textmining

def termdocumentmatrix_example():
    path = "/Users/franciscojavierarceo/MyPrograms/Python/"
    os.chdir(path)
    # Create some very short sample documents
    doc1 = 'John and Bob are brothers.'
    doc2 = 'John went to the store. The store was closed.'
    doc3 = 'Bob went to the store too.'
    # Initialize class to create term-document matrix
    tdm = textmining.TermDocumentMatrix()
    # Add the documents
    tdm.add_doc(doc1)
    tdm.add_doc(doc2)
    tdm.add_doc(doc3)
    # Write out the matrix to a csv file. Note that setting cutoff=1 means
    # that words which appear in 1 or more documents will be included in
    # the output (i.e. every word will appear in the output). The default
    # for cutoff is 2, since we usually aren't interested in words which
    # appear in a single document. For this example we want to see all
    # words however, hence cutoff=1.
    tdm.write_csv('matrix.csv', cutoff=1)
    print tdm
    # Instead of writing out the matrix you can also access its rows directly.
    # Let's print them to the screen.
    for row in tdm.rows(cutoff=1):
        print row

# This actually calls the function        
if __name__ == "__main__":
    termdocumentmatrix_example()