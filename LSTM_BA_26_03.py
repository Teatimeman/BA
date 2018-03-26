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
import re

from python_speech_features import mfcc

import sklearn
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile , join

# needed library for display
#import matplotlib 
#matplotlib.use("nbagg")

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
# Operating system commands


frame_length = 0.025
frame_step = 0.01

n_mfcc = 40

# rnn_size = Weights = input_dim + output_dim



## TODO: schneide in kleinere Trainingsdaten

def getSegments(partWaveFile,w_tier):    
    number = 0
    substring = -1
    while(re.match("\\d",partWaveFile[substring]) != None):
        number = number + int(partWaveFile[substring])*(10**(-substring-1))
        substring = substring - 1
    
    offset = (w_tier[number+1].bounds()[1] - w_tier[number+1].bounds()[0])/2
    start = (w_tier[number].bounds()[0]-offset)
    newDuration = (w_tier[number+4].bounds()[1] + offset)  - start  
    
    xmin1 = w_tier[number].bounds()[0] - start
    
    xmax1 = w_tier[number].bounds()[1] - start    

    xmin2 = w_tier[number+2].bounds()[0] - start
    xmax2 = w_tier[number+2].bounds()[1] - start
    xmin3 = w_tier[number+4].bounds()[0] - start
    xmax3 = w_tier[number+4].bounds()[1] - start
    return xmin1,xmax1,xmin2,xmax2,xmin3,xmax3,newDuration


def get_mfcc(audioSignal):
    signal, samplerate  = librosa.load(audioSignal,sr = None)
    mfcc_signal = mfcc(signal,samplerate,nfilt=40,numcep=40)
    return mfcc_signal

def get_label(audioSignal):
    T = textgrid.TextGrid();
    T.read("004_wav.TextGrid")
    w_tier = T.getFirst("Vokale").intervals
    mfcc_raw = get_mfcc(audioSignal)
    time_mark = 0.025/2 + 0.01*np.arange(0, mfcc_raw.shape[0])
    time_mark = time_mark.astype('float32')

    ## Generieren eines neuen Outputs/Labeling
    xmin1,xmax1,xmin2,xmax2,xmin3,xmax3, new_duration = getSegments(audioSignal,w_tier)
    words_raw = []
    for t in time_mark:       
        if t > xmin1 and t <= xmax1:
           words_raw.append(1.0)
        elif t > xmin2 and t <= xmax2:
           words_raw.append(1.0)
        elif t > xmin3 and t <= xmax3:
           words_raw.append(1.0)
        else: words_raw.append(0.0) 
 
    #words_raw = [None,None,None,..,1.WORT,1.WORT,1.WORT,...]
    words_label = np.asarray(words_raw, dtype = float)
    return words_label

def get_data(folder):
    
    signals = [join(folder,s) for s in listdir(folder) if isfile(join(folder,s))] ##Liste der Signale     
    signal_mfccs = [get_mfcc(s) for s in signals]
    signal_labels = [get_label(s) for s in signals] ## den output vector für alle signale erzeugen
    
    
    return signal_mfccs , signal_labels 

def prepare_data(folder):
    x_data, y_data  = get_data(folder)
    
    train_data = [(x_data[i],y_data[i]) for i in range(0, len(x_data))]
    x_train = np.asarray([pair[0] for pair in train_data])
    y_train = np.asarray([pair[1] for pair in train_data])
    
    x_input = [x.reshape((1,x.shape[0],x.shape[1])) for x in x_train]
    y_input = [y.reshape((1,y.shape[0])) for y in y_train]
    
    return x_input, y_input
 
x = tf.placeholder(tf.float32, [None, None, n_mfcc])
y = tf.placeholder(tf.float32, [None, None])

rnn_size = 512

#saver = tf.train.Saver()

def recurrent_neural_network():
    # x = tf.reshape(x, [-1, chunk_size])
    # x = tf.split(x, n_chunks)
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,1])),
             'biases':tf.Variable(tf.random_normal([1]))}
    
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,reuse=None)
#    with tf.variable_scope('LSTM1'):
    
    outputs, states = rnn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    
    #output_shaped = tf.reshape(output, [ -1, words_label.shape])
    print("Output-Shape: ;" +output.shape)
    return output

def train_neural_network(learning_rate = 0.01, batch_size=1 ,hm_epochs=5):
    
    
    x_data, y_data = prepare_data("004_slices")
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    

    prediction = recurrent_neural_network()
    # oder netz ist nicht mit der loss funktion verbunden
    ## sigmoid konstante C um 0 und 1 auszugleichen 
    
    #cost = tf.reduce_mean(tf.squared_difference(prediction[1:1672], y))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = prediction))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
#        print("shape", x_train.shape, y_train.shape)
#        saver.restore(sess, "/tmp/model.ckpt")
        
        train_dict = {x: x_train, y: y_train}
        test_dict = {x: x_test, y: y_test}
        
#        print("Initial training loss: " + str(sess.run(cost, train_dict)))
#        print("Initial test loss: " + str(sess.run(cost, test_dict)))
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            epoch_data = list(zip(x_train, y_train))
            np.random.shuffle(epoch_data)
            
            for x_sample,y_sample in epoch_data:
#                epoch_x, epoch_y = speech_data.next_batch(batch_size)
#                epoch_x = x_sample.reshape((1,x_sample.shape[0],x_sample.shape[1]))
#                epoch_y = y_sample.reshape((1,y_sample.shape[0]))
                epoch_x = x_sample
                epoch_y = y_sample
                _, c= sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            average_loss = epoch_loss / len(epoch_data)
            print('Epoch', str(epoch), 'completed out of',hm_epochs,
                  'loss:', str(epoch_loss),
                  'average loss', str(average_loss))
            

#            pred_out = sess.run(prediction, feed_dict={x_train: y_train})
#            pred_out = np.argmax(pred_out, 1)
#            
#        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#
#        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#        
#        print('Accuracy:',accuracy.eval(train_dict))

train_neural_network()























