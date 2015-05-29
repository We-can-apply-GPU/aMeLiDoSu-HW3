#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import word2vec
import numpy as np
import theano.tensor as T
import theano
import pickle
import network
import sys

from settings import *
from train import build_model

def norm(t):
    return T.sqrt(T.sqr(t).sum())

def load_data():
    fin = open("data/testing_data.txt", "r")
    lines = []
    for line in fin:
        line = line[line.find(")")+2:-1]
        line = line.replace("[", "")
        line = line.replace("]", "")
        lines.append(line)
    return lines

def main():
    print("Loading data...")
    lines = load_data()
    w2v = word2vec.load("train_pro.txt.bin")

    print("Building model")
    output_layer = build_model(bi_directional = True)
    network.layers.set_all_param_values(output_layer, pickle.load(open("model/"+sys.argv[1])))

    seq = T.matrix('seq')
    normal = seq*100

    objective = network.objectives.Objective(output_layer, loss_function=network.objectives.mse)
    loss = objective.get_loss(normal, target=normal, training=False)
    get_loss = theano.function(inputs=[seq], outputs=loss)

    fout = open("ans.csv", "w")
    index = 0
    loss = [0] * 5
    for line in lines:
        tmp = []
        for word in line:
            if word in w2v.vocab:
                tmp.append(w2v[word])
        _, loss[index%5], _ = get_loss(np.array(tmp, dtype="float32"))
        index += 1
        if index%5==0:
            print("{},{}".format(index/5, chr(np.argmin(loss)+ord('a'))))
            print("{},{}".format(index/5, chr(np.argmin(loss)+ord('a'))), file=fout)

if __name__ == "__main__":
    main()
