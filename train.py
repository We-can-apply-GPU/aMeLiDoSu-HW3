#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import numpy as np
import theano
import theano.tensor as T
import network
import time
import os
import random
import word2vec
import pickle

from settings import *

def norm(t):
    return T.sqrt(T.sqr(t).sum())

#def cross_entropy(a,b):
    #return ((-1) * T.log(1 - a)*(1-b) + b*T.log(a))

def load_data():
    fin = open("data/allsen", "r")
    data = []
    cnt = 0
    for line in fin:
        line = line[:-2]
        line = line.split(' ')
        """
        if "swear" not in line:
            continue
        """
        data.append(line)
    return np.array(data)

def build_model(bi_directional = False):

    if bi_directional:
        l_in = network.layers.InputLayer(
            shape=(None, WORD_2_VEC_FEATURES),name="InputLayer")

        l_rec_forward = network.layers.RecurrentLayer(
            l_in, num_units=NUM_HIDDEN_UNITS, trace_steps = 4, name="ForwardLayer")

        l_rec_backward = network.layers.RecurrentLayer(
            l_in, num_units=NUM_HIDDEN_UNITS, backwards=True, trace_steps = 4, name="BackwardLayer")

        l_sum = network.layers.SummingLayer(
            incomings = (l_rec_forward, l_rec_backward), f = 0.8, name="SummingLayer")

        l_out = network.layers.DenseLayer(
            incoming = l_sum, num_units = WORD_2_VEC_FEATURES, name="OutputProjection")

        """
        l_out = network.layers.OutputLayer(
            l_rec_combined, num_units=WORD_2_VEC_FEATURES, name="OutputLayer")
        """
    else:
        l_in = network.layers.InputLayer(
                shape=(None, WORD_2_VEC_FEATURES),name="InputLayer")

        l_out = network.layers.RecurrentLayer(
                l_in, num_units=NUM_HIDDEN_UNITS,name="RecurrentLayer")

        """
        l_out = network.layers.OutputLayer(
                l_recurrent, num_units=WORD_2_VEC_FEATURES,name="OuptutLayer")
        """

    return l_out

def create_iter_functions(output_layer, learning_rate=LEARNING_RATE,momentum=MOMENTUM):

    seq = T.matrix('seq')
    objective = network.objectives.Objective(output_layer, loss_function=network.objectives.mse)

    loss = objective.get_loss(seq, target=seq)
    #accuracy =T.mean(T.eq(output_layer.get_output(X_batch,deterministic = True),Y_batch),dtype = theano.config.floatX)
    #compare_shape = (BATCH_SIZE*NGRAMS*WORD_2_VEC_FEATURES,)


    XX = T.flatten(output_layer.get_output(seq))
    YY = T.flatten(seq)
    accuracy = T.dot(XX,YY)/(norm(XX) * norm(YY))

    #XXX = XX[WORD_2_VEC_FEATURES*(NGRAMS-1)+1:]
    #YYY = YY[WORD_2_VEC_FEATURES*(NGRAMS-1)+1:]
    #pred = T.argmax(output_layer.get_output(X_batch, deterministic=True), axis=1)
    #errorRate = T.norm(output_layer.get_output(X_batch,deterministic = True) - Y_batch)

    all_params = network.layers.get_all_params(output_layer)
    updates = network.updates.rmsprop(loss, all_params, LEARNING_RATE, MOMENTUM)

    iter_train = theano.function(inputs=[seq], outputs=loss, updates=updates)
    iter_valid = theano.function(inputs=[seq], outputs=[loss, accuracy])

    return {'train': iter_train, 'valid': iter_valid}

def main():
    print("Loading data...")
    data = load_data()
    w2v = word2vec.load("train_pro.txt.bin")

    print("Building model")
    output_layer = build_model(bi_directional = True)

    print ('Creating iter functions')
    iter_funcs = create_iter_functions(output_layer)

    print("Training")
    try:
        for epoch in range(NUM_EPOCHS):
            now = time.time()
            batch_train_losses = []
            for cnt in range(100):
                seq = data[random.randrange(0, len(data))]
                tmp = []
                for word in seq:
                    if word in w2v.vocab:
                        tmp.append(w2v[word])
                batch_train_loss = iter_funcs['train'](np.array(tmp, dtype="float32"))
                batch_train_losses.append(batch_train_loss)
            avg_train_loss = np.mean(batch_train_losses)
            print("Epoch {} of {} took {:.3f}s".format(epoch+1, NUM_EPOCHS, time.time() - now))
            print("  training loss:\t\t{:.6f}".format(avg_train_loss))
            if epoch % 10 == 0:
                batch_valid_accus = []
                batch_valid_losses = []
                for cnt in range(100):
                    seq = data[cnt]
                    tmp = []
                    for word in seq:
                        if word in w2v.vocab:
                            tmp.append(w2v[word])
                    batch_valid_loss, batch_valid_accu = iter_funcs['valid'](np.array(tmp, dtype="float32"))
                    batch_valid_losses.append(batch_valid_loss)
                    batch_valid_accus.append(batch_valid_accu)
                avg_valid_loss = np.mean(batch_valid_losses)
                avg_valid_accu = np.mean(batch_valid_accus)
                print("--validation loss:\t\t{:.6f}".format(avg_valid_loss))
                print("--validation accuracy:\t\t{:.4f} %".format(avg_valid_accu))
                #write model
                fout = open("model/{:.2f}".format(avg_valid_accu * 100), "w")
                pickle.dump(network.layers.get_all_param_values(output_layer), fout)
                now = time.time()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()

