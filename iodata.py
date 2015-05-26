#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: iodata.py
Description: produce in, out data for train
"""

import extract
import sentence
from settings import *
import numpy  as np
indatalist = []
outdatalist = []
inlist = []
outlist = []
def iodata():
    _list = sentence.getsen('swear')
    dic = extract.extract(100, 'train_pro.txt', 'true')
    for i in range(0, len(_list)):
        inlist = []
        outlist = []
        for j in range(0, len(_list[i])):
            _str = _list[i][j].split()
            for k in range(0, len(_str)):
                if(len(_str)<=NGRAMS or (len(_str)-k) <= NGRAMS):
                    continue
                else:
                    if(dic.get(_str[k]) == None or dic.get(_str[k+1]) == None 
                            or dic.get(_str[k+2])== None 
                            or dic.get(_str[k+3])== None
                            or dic.get(_str[k+4]) == None):
                        continue
                    else:
                        if((len(_str)-k)>NGRAMS):
                            indata = [dic[_str[k]],dic[_str[k+1]],dic[_str[k+2]],dic[_str[k+3]]]
                            outdata = [dic[_str[k+1]],dic[_str[k+2]],dic[_str[k+3]],dic[_str[k+4]]]
                            #indata = [_str[k+0],_str[k+1],_str[k+2],_str[k+3]]
                            #outdata = [_str[k+1],_str[k+2],_str[k+3],_str[k+4]]
                            #print(outdata)
                            inlist.append(indata)
                            outlist.append(outdata)
        index = 0
        while(len(inlist)<BATCH_SIZE):
            inlist.append(inlist[index])
            outlist.append(outlist[index])
            index = index + 1
        #print(len(inlist))
        indatalist.extend(inlist)
        outdatalist.extend(outlist)
        #pp = np.array(indatalist)
        #print(pp.shape)
        #print(indatalist)
    return indatalist, outdatalist
def for_pre(_str):
    dic = extract.extract(100, 'train_pro.txt', 'true')
    for i in range(0, len(_str)):
        if(len(_str)>=NGRAMS and (len(_str)-i) >= NGRAMS):
            if(dic.get(_str[i]) == None or dic.get(_str[i+1]) == None
               or dic.get(_str[i+2])== None):
                continue
            else:
                if((len(_str)-i)>=NGRAMS):
                    indata = [dic[_str[i]],dic[_str[i+1]],dic[_str[i+2]]]
                    inlist.append(indata)
                    #outlist.append(outdata)
    return inlist
    #outdatalist.extend(outlist)


if __name__=="__main__":
    iodata()
