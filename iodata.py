#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: iodata.py
Description: produce in, out data for train
"""

import extract

str1 = "copyright laws are changing all over the world be sure to check the copyright laws for your country before posting these files"
str2 = "please take a look at the important information in this header"
str3 = "we encourage you to keep this file on your own disk keeping an electronic path open for the next readers "
str4 = "do not remove this"
ss0 = [str1.split(),str2.split()]
ss1 = [str3.split(),str4.split()]
sslist = [ss0,ss1]
Ngram = 3
batchsize = 30
indatalist = []
outdatalist = []

def iodata():
    """
    for i in range(0,len(words)):
        print(i)
        print(words[i])
    """
    #dic = extract.extract(100, 'train_pro.txt', 'true')
    for i in range(0, len(sslist)):
        inlist = []
        outlist = []
        for j in range(0, len(sslist[i])):
            for k in range(0, len(sslist[i][j])):
                if(len(sslist[i][j])>Ngram):
                    if((len(sslist[i][j])-k)>Ngram):
                        #indata = [dic[sslist[i][j][k]],dic[sslist[i][j][k+1]],dic[sslist[i][j][k+2]]]
                        #outdata = [dic[sslist[i][j][k+1]],dic[sslist[i][j][k+2]],dic[sslist[i][j][k+3]]]
                        indata = [sslist[i][j][k+0],sslist[i][j][k+1],sslist[i][j][k+2]]
                        outdata = [sslist[i][j][k+1],sslist[i][j][k+2],sslist[i][j][k+3]]
                        inlist.append(indata)
                        outlist.append(outdata)
        index = 0
        while(len(inlist)<batchsize):
            inlist.append(inlist[index])
            outlist.append(outlist[index])
            index = index + 1
        for l in range(0, len(inlist)):
            print(inlist[l])
        indatalist.append(inlist)
        outdatalist.append(outlist)
    print(len(indatalist))                  
    print(len(outdatalist))                  

if __name__=="__main__":
    iodata()
