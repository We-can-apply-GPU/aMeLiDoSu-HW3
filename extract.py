#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
File: extract.py
Description: feature extraction
"""

import word2vec
import numpy as np

dim = 100
data = 'train_pro.txt'

def extract(dim, data, trained):
    if(not trained):
        word2vec.word2phrase(data, data+'-phrases', verbose=True)
        word2vec.word2vec(data+'-phrases', data+'.bin', size=dim, verbose=True)
    model = word2vec.load(data+'.bin')
    keys = model.vocab
    features = model.vectors
    dic = dict(zip(keys,features))
    print(len(dic))
    return dic    
    
if __name__=="__main__":
    extract(dim, data, true)
