#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import pickle
import network
import time
import theano
import os
import theano.tensor as T
import sys
from settings import *
import itertools
import iodata

QUESTION_SIZE = 1040
NUM_CHOICES = 5
def norm(vec):
    return T.sqrt(T.sqr(vec).sum())

def cos_dis(x,y):
    #only get last word
    ret = T.dot(x,y)/(norm(x)*norm(y))
    return ret

def calculate_cos_dis(ix,iy):
    cos_dis_ls = T.zeros([NUM_CHOICES])
    x = T.reshape(ix,(NUM_CHOICES,NGRAMS*WORD_2_VEC_FEATURES))
    y = T.reshape(iy,(NUM_CHOICES,WORD_2_VEC_FEATURES))
    
    for index in range(NUM_CHOICES):
        vx = x[index][WORD_2_VEC_FEATURES*(NGRAMS-1)+1:]
        vy = y[index]
        cos_dis_ls = T.set_subtensor(cos_dis_ls[index],cos_dis(vx,vy))
    cos_dis_ls = cos_dis_ls.dimshuffle('x',0)
    #print("YAYAYAY")
    return cos_dis_ls
    #return T.reshape(cos_dis_ls,(1,NUM_CHOICES))

def build_model(bi_directional = False):

    if bi_directional:
        l_in = network.layers.InputLayer(
                shape=(NUM_CHOICES,NGRAMS,WORD_2_VEC_FEATURES),name="InputLayer")
        
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
                shape=(NUM_CHOICES,NGRAMS,WORD_2_VEC_FEATURES),name="InputLayer")
        l_recurrent = network.layers.RecurrentLayer(
                l_in, num_units=NUM_HIDDEN_UNITS,name="RecurrentLayer")

        l_out = network.layers.OutputLayer(
                l_recurrent, num_units=WORD_2_VEC_FEATURES,name="OuptutLayer")

    return l_out

def main():
    pars = "model/4GRAM_BI/76.69"
    print("Loading data...")
    (feats_in,feats_out) = iodata.iodata_forPre()
    feats_in = np.array(feats_in).astype(theano.config.floatX)
    print("{}".format((QUESTION_SIZE*(NGRAMS+1)*NUM_CHOICES,1,WORD_2_VEC_FEATURES)))
    feats_out = np.array(feats_out).astype(theano.config.floatX).reshape((QUESTION_SIZE*(NGRAMS+1)*NUM_CHOICES,1,WORD_2_VEC_FEATURES))
    #print(feats_out.shape)
    #print(feats_in)
    #print(lenfeats_out)
    output_layer = build_model(bi_directional = True)
    network.layers.set_all_param_values(output_layer, pickle.load(open(pars, "r")))
    x = T.tensor3('x', dtype=theano.config.floatX)
    y  =T.tensor3('y',dtype = theano.config.floatX)  
    cos_distance_ls = np.zeros((QUESTION_SIZE,NUM_CHOICES))
    predict = theano.function([x,y],calculate_cos_dis(output_layer.get_output(x,deterministic=True),y),on_unused_input='ignore')

    for index in range(QUESTION_SIZE):
        try:
            print(feats_in[(index)*NUM_CHOICES:(index+1)*NUM_CHOICES].shape)
            print(feats_out[(index)*NUM_CHOICES:(index+1)*NUM_CHOICES].shape)
            pred  = predict(feats_in[(index)*NUM_CHOICES:(index+1)*NUM_CHOICES],feats_out[(index)*NUM_CHOICES:(index+1)*NUM_CHOICES])
            print("OHOHOH")
        except RuntimeError:
            pass
        cos_distance_ls[index,:] = cos_distance_ls[index,:] + pred

if __name__ == '__main__':
    main()
