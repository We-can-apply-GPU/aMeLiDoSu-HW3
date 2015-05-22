#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: iodata.py
Description: produce in, out data for train
"""

import extract
import sentence
from settings import *

indatalist = []
outdatalist = []

def iodata():
    _list = sentence.getsen('swear')
    dic = extract.extract(100, 'train_pro.txt', 'true')
    for i in range(0, len(_list)):
        inlist = []
        outlist = []
        for j in range(0, len(_list[i])):
            _str = _list[i][j].split()
            for k in range(0, len(_str)):
                if(len(_str)>NGRAMS):
                    if((len(_str)-k)>NGRAMS):
                        #indata = [dic[_str[k]],dic[_str[k+1]],dic[_str[k+2]]]
                        #outdata = [dic[_str[k+1]],dic[_str[k+2]],dic[_str[k+3]]]
                        indata = [_str[k+0],_str[k+1],_str[k+2]]
                        outdata = [_str[k+1],_str[k+2],_str[k+3]]
                        inlist.append(indata)
                        outlist.append(outdata)
        index = 0
        while(len(inlist)<BATCH_SIZE):
            inlist.append(inlist[index])
            outlist.append(outlist[index])
            index = index + 1
        indatalist.append(inlist)
        outdatalist.append(outlist)
    return indatalist, outdatalist

if __name__=="__main__":
    iodata()