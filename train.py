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
import itertools
import iodata
from settings import *


#impotr util
import pickle
#import random
def norm(t):
    return T.sqrt(T.sqr(t).sum())
#def cross_entropy(a,b):
    #return ((-1) * T.log(1 - a)*(1-b) + b*T.log(a))
def load_data():

    X_train ,Y_train = iodata.iodata()
    X_train = np.array(X_train).astype(np.float32)
    Y_train = np.array(Y_train).astype(np.float32)
    return dict(
            X_train = theano.shared(X_train),
            Y_train = theano.shared(Y_train),
            num_train=len(X_train))

def build_model(bi_directional = False):

    if bi_directional:
        l_in = network.layers.InputLayer(
                shape=(BATCH_SIZE,NGRAMS,WORD_2_VEC_FEATURES),name="InputLayer")
        
        l_rec_forward = network.layers.RecurrentLayer(
                l_in,num_units=NUM_HIDDEN_UNITS,name="ForwardLayer")

        l_rec_backward = network.layers.RecurrentLayer(
                l_in,num_units=NUM_HIDDEN_UNITS,backwards=True,name="BackwardLayer")

        l_rec_combined = network.layers.ElemwiseSumLayer(
                incomings = (l_rec_forward, l_rec_backward),name="SummingLayer")

        l_out = network.layers.OutputLayer(
                l_rec_combined,num_units=WORD_2_VEC_FEATURES,name="OutputLayer")
    else:
        l_in = network.layers.InputLayer(
                shape=(BATCH_SIZE,NGRAMS,WORD_2_VEC_FEATURES),name="InputLayer")
        l_recurrent = network.layers.RecurrentLayer(
                l_in, num_units=NUM_HIDDEN_UNITS,name="RecurrentLayer")

        #l_reshape1 = network.layers.ReshapeLayer(l_recurrent,shape=(BATCH_SIZE,NGRAMS*NUM_HIDDEN_UNITS))
        #l_middle = network.layers.DenseLayer(l_recurrent,num_units = NUM_HIDDEN_UNITS,nonlinearity = network.nonlinearities.rectify)

        #l_reshape2 = network.layers.ReshapeLayer(l_middle,shape=(BATCH_SIZE,NGRAMS,NUM_HIDDEN_UNITS))
        l_out = network.layers.OutputLayer(
                l_recurrent, num_units=WORD_2_VEC_FEATURES,name="OuptutLayer")

    return l_out

def create_iter_functions(data, output_layer, batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE,momentum=MOMENTUM):

    batch_index = T.iscalar('batch_index')
    X_batch = T.tensor3('x')
    Y_batch = T.tensor3('y')
    batch_slice = slice(batch_index * BATCH_SIZE,(batch_index+1) * BATCH_SIZE)
    #print("{}     {}".format(batch_index * BATCH_SIZE,(batch_index+1) * BATCH_SIZE))
    #print(batch_slice)

    objective = network.objectives.Objective(output_layer,
            loss_function=network.objectives.mse)

    loss_train = objective.get_loss(X_batch, target=Y_batch)
    loss_eval = objective.get_loss(X_batch, target=Y_batch, deterministic=True)
    #accuracy =T.mean(T.eq(output_layer.get_output(X_batch,deterministic = True),Y_batch),dtype = theano.config.floatX)
    compare_shape = (BATCH_SIZE*NGRAMS*WORD_2_VEC_FEATURES,)
    XX = T.reshape(output_layer.get_output(X_batch,deterministic = True),compare_shape)
    YY = T.reshape(Y_batch,compare_shape)
    accuracy = T.dot(XX,YY)/(norm(XX) * norm(YY))
    #pred = T.argmax(output_layer.get_output(X_batch, deterministic=True), axis=1)
    #errorRate = T.norm(output_layer.get_output(X_batch,deterministic = True) - Y_batch)

    all_params = network.layers.get_all_params(output_layer)
    updates = network.updates.rmsprop(loss_train, all_params, LEARNING_RATE, MOMENTUM)

    iter_train = theano.function(
            [batch_index], loss_train,
            updates=updates,
            givens={
                X_batch: data['X_train'][batch_slice],
                Y_batch: data['Y_train'][batch_slice],
                },
            )

    iter_valid = theano.function(
            [batch_index], [loss_eval, accuracy],
            givens={
                X_batch: data['X_train'][batch_slice],
                Y_batch: data['Y_train'][batch_slice],
                },
            )

    return dict(
            train=iter_train,
            valid=iter_valid)

def main():
    print("Loading data...")
    data = load_data()

    print("Building model and compile theano...")
    print(data['num_train'])
    output_layer = build_model(bi_directional = True)

    print ('Creating iter functions')
    iter_funcs = create_iter_functions(data, output_layer)

    print("Training")
    now = time.time()
    try:
        for epoch in range(NUM_EPOCHS):
            num_batches_train = data['num_train'] // BATCH_SIZE
            batch_train_losses = []
            for b in range(num_batches_train):
                batch_train_loss = iter_funcs['train'](b)
                batch_train_losses.append(batch_train_loss)
            avg_train_loss = np.mean(batch_train_losses)
            print("Epoch {} of {} took {:.3f}s".format(epoch+1, NUM_EPOCHS, time.time() - now))
            print("  training loss:\t\t{:.6f}".format(avg_train_loss))
            if epoch % 10 == 0:
                batch_valid_accus = []
                batch_valid_losses = []
                for b in range(num_batches_train):
                    batch_valid_loss, batch_valid_accu = iter_funcs['valid'](b)
                    batch_valid_losses.append(batch_valid_loss)
                    batch_valid_accus.append(batch_valid_accu)
                avg_valid_loss = np.mean(batch_valid_losses)
                avg_valid_accu = np.mean(batch_valid_accus)
                print("--validation loss:\t\t{:.2f}".format(avg_valid_loss))
                print("--validation accuracy:\t\t{:.2f} %".format(avg_valid_accu))
                #write model
                fout = open("model/5d/{:.2f}".format(avg_valid_accu * 100), "w")
                pickle.dump(network.layers.get_all_param_values(output_layer), fout)
                now = time.time()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()

