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
inlist1 = []
inlist2 = []
inlist3 = []
inlist4 = []
inlist5 = []
outlist1 = []
outlist2 = []
outlist3 = []
outlist4 = []
outlist5 = []
dic = extract.extract(100, 'train_pro.txt', 'true')
d = open('data/predic_data')
l = d.readlines();
z = np.zeros(100)

def iodata():
    _list = sentence.getsen('swear')
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
#def for_pre():
            #indata = [_str[-3], _str[-2], _str[-1], l[i*6+j+1].split()[0]]
    #print(inlist)
    #for i in range(0, len(_str)):
    #    if(len(_str)>=NGRAMS and (len(_str)-i) >= NGRAMS):
    #        if(dic.get(_str[i]) == None or dic.get(_str[i+1]) == None
    #           or dic.get(_str[i+2])== None):
    #            continue
    #        else:
    #            if((len(_str)-i)>=NGRAMS):
    #                indata = [dic[_str[i]],dic[_str[i+1]],dic[_str[i+2]]]
    #                inlist.append(indata)
    #                #outlist.append(outdata)
    #return inlist
    #outdatalist.extend(outlist)

def for_pre():
    for i in range (0, 1040):
        _str = l[i*6].split()
        for j in range (0,5):
            if dic.get(_str[-4]) == None:
                dic[_str[-4]] = z
            if dic.get(_str[-3]) == None:
                dic[_str[-3]] = z
            if dic.get(_str[-2]) == None:
                dic[_str[-2]] = z
            if dic.get(_str[-1]) == None:
                dic[_str[-1]] = z
            if dic.get(l[i*6+j+1].split()[0]) == None:
                dic[l[i*6+j+1].split()[0]] = z
            if dic.get(_str[0]) == None:
                dic[_str[0]] = z
            if dic.get(_str[1]) == None:
                dic[_str[1]] = z
            if dic.get(_str[2]) == None:
                dic[_str[2]] = z
            if dic.get(_str[3]) == None:
                dic[_str[3]] = z
            indata1 = [dic[_str[-4]], dic[_str[-3]], dic[_str[-2]], dic[_str[-1]]]
            indata2 = [dic[_str[-3]], dic[_str[-2]], dic[_str[-1]], dic[l[i*6+j+1].split()[0]]]
            indata3 = [dic[_str[-2]], dic[_str[-1]], dic[l[i*6+j+1].split()[0]], dic[_str[0]]]
            indata4 = [dic[_str[-1]], dic[l[i*6+j+1].split()[0]], dic[_str[0]], dic[_str[1]]]
            indata5 = [dic[l[i*6+j+1].split()[0]], dic[_str[0]], dic[_str[1]], dic[_str[2]]]
            #indata = [_str[-3], _str[-2], _str[-1], l[i*6+j+1].split()[0]]
            outdata1 = [dic[l[i*6+j+1].split()[0]]]
            outdata2 = [dic[_str[0]]]
            outdata3 = [dic[_str[1]]]
            outdata4 = [dic[_str[2]]]
            outdata5 = [dic[_str[3]]]
            inlist1.append(indata1)
            inlist2.append(indata2)
            inlist3.append(indata3)
            inlist4.append(indata4)
            inlist5.append(indata5)
            outlist1.append(outdata1)
            outlist2.append(outdata2)
            outlist3.append(outdata3)
            outlist4.append(outdata4)
            outlist5.append(outdata5)
    indatalist.extend(inlist1)
    indatalist.extend(inlist2)
    indatalist.extend(inlist3)
    indatalist.extend(inlist4)
    indatalist.extend(inlist5)
    outdatalist.extend(outlist1)
    outdatalist.extend(outlist2)
    outdatalist.extend(outlist3)
    outdatalist.extend(outlist4)
    outdatalist.extend(outlist5)
    return indatalist, outdatalist

if __name__=="__main__":
    for_pre()
