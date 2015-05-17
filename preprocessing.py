#!/usr/bin/env python
# _*_ coding: utf-8 -*-
"""
File: preprocessing.py
Description: handle training data, delete <S> and change it to a line in train_pro.txt
"""
import os
import sys

def iofile():
    _str = ''
    if(sys.argv[2]=='-a'):
        pwd = os.path.basename(sys.argv[1])
        for filename in os.listdir(sys.argv[1]):
            print(filename)
            os.system('cat ' + sys.argv[1] + '/' + filename + ' | ./preprocessing.sh > train.txt') 
            art = open('train.txt','r')
            for line in art:
                #print(line[5:-5])
                if(isinstance(line[5:-5], str)):
                    _str = _str + line[5:-5]
                else:
                    _str = _str + unicode(line[5:-5], errors = 'ignore')
    else:
        print(sys.argv[2])
        os.system('cat ' + sys.argv[1] + '/' + sys.argv[2] + ' | ./preprocessing.sh > train.txt') 
        art = open('train.txt','r')
        _str = ''
        for line in art:
            if(isinstance(line[5:-5], str)):
                _str = _str + line[5:-5]
            else:
                _str = _str + unicode(line[5:-5], errors = 'ignore')
    art = open('train_pro.txt','w')
    art.write(_str)

if __name__=="__main__":
    iofile()
