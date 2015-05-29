#!/usr/bin/env python
from __future__ import print_function
from pygoogle import pygoogle
"""
fout = open("blank.txt", "w")
for i in range(201, 1041):
    print("{},b".format(i), file=fout)
"""
f = open("data/testing_data.txt", "r")
fout = open("ans.txt", "w")
index = 0
options = [0] * 5
for line in f:
    line = line[line.find(")")+2:-1]
    line = line.replace("[", "")
    line = line.replace("]","")
    line = "\""+line+"\""
    #print(line)
    options[index%5] = pygoogle(line).get_result_count()
    index += 1
    if index%5==0:
        #print(options)
        max_value=-1
        max_index=-1
        for i,a in enumerate(options):
            if a>max_value:
                max_value = a
                max_index = i
        print("{},{}".format(index/5, chr(max_index+ord('a'))))
        print("{},{}".format(index/5, chr(max_index+ord('a'))), file=fout)
