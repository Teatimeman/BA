#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:56:10 2017

@author: teatimeman
"""
import numpy as np
import tensorflow as tf

import librosa
#import librosa.display

from python_speech_features import mfcc

## needed library for display
#import matplotlib 
#matplotlib.use("nbagg")
#
#import matplotlib.pyplot as plt
## Style for display
#import matplotlib.style as ms
#
## IPython gives us an audio widget for playback
#from IPython.display import Audio

# Parsen Textgrid
import textgrid

# Netz modellierung
from tensorflow.python.ops import rnn, rnn_cell
# Operating system Kommandos 
import os


frame_length = 0.025
frame_step = 0.01

n_mfcc = 40
n_examples = 1 
# rnn_size = Weights = input_dim + output_dim

sig_orig, sr_orig = librosa.load("004.wav.wav",sr= None)
    
print("sig: ",len(sig_orig)," sr: ",sr_orig)

##sampling rate
#sr= 16000
#
#sig = librosa.resample(sig_orig,sr_orig,sr)

mfcc_raw = mfcc(sig_orig,sr_orig,nfilt=40,numcep=40)

print(mfcc_raw.shape)



#def length(sequence):
#  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
#  length = tf.reduce_sum(used, 1)
#  length = tf.cast(length, tf.int32)
#  return length
#
#max_length = 100
#frame_size = 64
#num_hidden = 200
#
#sequence = tf.placeholder(tf.float32, [None, max_length, frame_size])
#
#output, state = tf.nn.dynamic_rnn(
#    tf.contrib.rnn.GRUCell(num_hidden),
#    sequence,
#    dtype=tf.float32,
#    sequence_length=length(sequence),
#)
#
#print("sequence_length: ", sequence_length)
#

T = textgrid.TextGrid();
T.read("004_wav.TextGrid")

w_tier = T.getFirst("Vokale").intervals

#print("w_tier", w_tier)

time_mark = 0.025/2 + 0.01*np.arange(0, mfcc_raw.shape[0])
time_mark = time_mark.astype('float32')

#print("time_Mark", time_mark)
#print("time_mark",time_mark.shape)
#
#k1 = np.arange(0,mfcc_raw.shape[0])
#print("k1",k1)
#print("k1",k1.shape)
#
#k2 = 0.01*np.arange(0,mfcc_raw.shape[0])
#print("k2",k2)
#print("k2",k2.shape)

#
#
#for t in time_mark:
#    print("t",t)

words_raw = []
for t in time_mark:
    for ival in range(len(w_tier)):        
        if t > w_tier[ival].bounds()[0] and t <= w_tier[ival].bounds()[1]:
           words_raw.append(w_tier[ival].mark)


words_list = list(set(words_raw)) # unique word list
words_idx = {w: i for i, w in enumerate(words_list)}
words_data = [words_idx[w] for w in words_raw]
words_data_onehot = tf.one_hot(words_data,
                              depth = len(words_list),
                              on_value = 1.,
                              off_value = 0.,
                              axis = 1,
                              dtype=tf.float32)                


with tf.Session() as sess: # convert from Tensor to numpy array
    words_label = words_data_onehot.eval()

#print('words_list:',words_list)
#print('output dimension:',words_label.shape)
#
##print("words_raw", words_raw)
#print("words_list", words_list)
#print("words_idx", words_idx)
#print("words_data", words_data)
#print("words_data_onehot",words_data_onehot)

#Batchsize * sequence length * input dimension (coefficients) 
#mfcc sollte 40 sein

x = tf.placeholder(tf.float32, [None, None, n_mfcc])
y = tf.placeholder(tf.float32, [None, None, None])

rnn_size = 512

#
#weights =  {
#                'wout' : tf.Variable(tf.random_normal([rnn_size, words_label.shape[1]]))
#    }
#bias =  {
#                'bout' : tf.Variable(tf.random_normal([words_label.shape[1]]))
#    }

def recurrent_neural_network():
    # x = tf.reshape(x, [-1, chunk_size])
    # x = tf.split(x, n_chunks)
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, words_label.shape[1]])),
             'biases':tf.Variable(tf.random_normal([words_label.shape[1]]))}

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,forget_bias=1.0, reuse=True)
#    with tf.variable_scope('LSTM1'):
    
    outputs, states = rnn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    
    #output_shaped = tf.reshape(output, [ -1, words_label.shape])
 
    return output

def train_neural_network(learning_rate = 0.01, batch_size=1 ,hm_epochs=5):

    prediction = recurrent_neural_network()
    #cost = tf.reduce_mean(tf.squared_difference(prediction[1:1672], y))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y[-1], logits = prediction))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
#            for _ in range(int(speech_data.num_examples/batch_size)):
            for _ in range(int(n_examples)):
#                epoch_x, epoch_y = speech_data.next_batch(batch_size)
                epoch_x = mfcc_raw.reshape((batch_size,mfcc_raw.shape[0],mfcc_raw.shape[1]))
                epoch_y = words_label.reshape((batch_size,words_label.shape[0],words_label.shape[1]))
                _, c= sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            average_loss = epoch_loss / n_examples    
            print('Epoch', str(epoch), 'completed out of',hm_epochs,
                  'loss:', str(epoch_loss),
                  'average loss', str(average_loss))
            epoch += 1
            
        pred_out = sess.run(prediction, feed_dict={x: epoch_x})
        pred_out = np.argmax(pred_out, 1)
            
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        test = []
        test.append(prediction)
        print(test)
        
#        print('Accuracy:',accuracy.eval({x:speech_data.reshape((-1, n_chunks, chunk_size)), y:speech_data.labels}))


train_neural_network()
























