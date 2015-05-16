#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import numpy as np
import theano
import theano.tensor as T
import lasagne # will be network later
import time
import os
import itertools

from RNN import RecurrentLayer, RecurrentSoftmaxLayer
import settings

#impotr util
#import pickle
#import random

def load_data():
	#get data 
	
        #X_train = np.array([data[1:] for data in fbank]).astype(theano.config.floatX)
        #Y_train = np.array([data[1] for data in label]).astype('int32')

	return dict(
                X_train = theano.shared(X_train),
                Y_train = theano.shared(Y_train),
		num_train=X_train.shape[0],
		input_dim=X_train.shape[1],
		output_dim=OUTPUT_DIM,
		)	

def build_model(bi_directional = False):
    
    if bi_direc:
	l_in = lasagne.layers.InputLayer(
                shape=(BATCH_SIZE,NGRAMS,WORD_2_VEC_FEATURES),name="InputLayer")

	l_rec_forward = RecurrentLayer(
                l_in,num_units=num_hidden_units,name="ForwardLayer") 

	l_rec_backward = RecurrentLayer(
                l_in,num_units=NUM_UNITS,backwards=True,name="BackwardLayer") 

	l_rec_combined = lasagne.layers.ElemwiseSumLayer(
	    incomings = (l_rec_forward, l_rec_backward),name="SummingLayer")

	l_out = RecurrentSoftmaxLayer(
                l_rec_combined,num_units=WORD_2_VEC_FEATURES,name="OutputLayer")
    else:
        l_in = lasagne.layers.InputLayer(
                shape=(BATCH_SIZE,WORD_2_VEC_FEATURES),name="InputLayer")
        l_recurrent = RecurrentLayer(
                l_in, num_units=NUM_UNITS,name="RecurrentLayer") 
        l_out = RecurrentSoftmaxLayer(
                l_recurrent, num_units=WORD_2_VEC_FEATURES,name="OuptutLayer")

    return l_out

def create_iter_functions(data, output_layer,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE,
                          momentum=MOMENTUM):
    batch_index = T.iscalar('batch_index')
    X_batch = T.matrix('x')
    y_batch = T.ivector('y')

    batch_slice = slice(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE)

    objective = lasagne.objectives.Objective(output_layer, loss_function=lasagne.objectives.categorical_crossentropy)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)

    pred = T.argmax(output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch))

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.rmsprop(loss_train, all_params, LEARNING_RATE, MOMENTUM)
    
    iter_train = theano.function(
      [batch_index], [loss_train, accuracy],
      updates=updates,
      givens={
        X_batch: data['X_train'][batch_slice],
        y_batch: data['y_train'][batch_slice],
        },
      )

    iter_valid = theano.function(
      [batch_index], [loss_eval, accuracy],
      givens={
        X_batch: data['X_valid'][batch_slice],
        y_batch: data['y_valid'][batch_slice],
        },
      )

    return dict(
      train=iter_train,
      valid=iter_valid,
      )
def main():
    print("Loading data...")
    data = load_data()
    print("Building model and compile theano...")
    print(data['num_train'])
    output_layer = build_model(bi_directional = False)
    print 'Creating iter functions'
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
        print("--validation accuracy:\t\t{:.2f} %".format(avg_valid_accu * 100))

        #write model
        fout = open("model/5d/{:.2f}".format(avg_valid_accu * 100), "w")
        pickle.dump(lasagne.layers.get_all_param_values(output_layer), fout)
        now = time.time()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
   main() 

