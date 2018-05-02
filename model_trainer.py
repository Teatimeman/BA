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
import os
from os import listdir
from os.path import isfile , isdir, join

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


frame_length = 0.0025
frame_step = 0.001
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
    mfcc_signal = mfcc(signal,samplerate,winlen=frame_length,winstep=frame_step,nfilt=40,numcep=40)
    return mfcc_signal

def get_label(audioSignal):
    T = textgrid.TextGrid();
    head,tail = os.path.split(audioSignal)
    number = tail[0:3]
    T.read("TextGrids/"+number+"_wav.TextGrid")
    w_tier = T.getFirst("Vokale").intervals
    mfcc_raw = get_mfcc(audioSignal)
 
    # time_mark = 0.025/2 + [0.01,0.02,..., 0.01*mfcc_raw.shape[0]]
    time_mark = frame_length/2 + frame_step*np.arange(0, mfcc_raw.shape[0])
    time_mark = time_mark.astype('float32')


    ## Generieren eines neuen Outputs/Labeling
    xmin1,xmax1,xmin2,xmax2,xmin3,xmax3, new_duration = getSegments(audioSignal,w_tier)
    words_raw = []
    for t in time_mark:       
        if t > xmin1 and t <= xmax1:
           words_raw.append([0.0,1.0])
        elif t > xmin2 and t <= xmax2:
           words_raw.append([0.0,1.0])
        elif t > xmin3 and t <= xmax3:
           words_raw.append([0.0,1.0])
        else: words_raw.append([1.0,0.0]) 
 
    #words_raw = [None,None,None,..,1.WORT,1.WORT,1.WORT,...]
    words_label = np.asarray(words_raw, dtype = float)
    return words_label

    

def get_data(folder):
    
    # Liste der files(Signale) im folder
    signals = []
    for root, directories, files in os.walk(folder):
        for filename in files:
            signals.append(join(root,filename))
    
    signal_mfccs = [get_mfcc(s) for s in signals]
    signal_labels = [get_label(s) for s in signals] ## den output vector für alle signale erzeugen
    
    
    return signal_mfccs , signal_labels 

def prepare_data(folder):
    x_data, y_data  = get_data(folder)
    
    train_data = [(x_data[i],y_data[i]) for i in range(0, len(x_data))]
    x_train = np.asarray([pair[0] for pair in train_data])
    y_train = np.asarray([pair[1] for pair in train_data])
    return x_train,y_train    
    
#    x_input = [s.reshape((1,s.shape[0],s.shape[1])) for s in x_train]
#    y_input = [s.reshape((1,s.shape[0],s.shape[1])) for s in y_train]
#    return x_input, y_input

    
# batch_Size, Sequence_length, n_mfcc
x = tf.placeholder(tf.float32, [None, None, n_mfcc], name = "x")
# batch_Size, Sequence_length_labels
y = tf.placeholder(tf.float32, [None, None,2], name = "y")

def recurrent_neural_network():
#     x = tf.reshape(x, [-1, chunk_size])
    # x = tf.split(x, n_chunks)
    rnn_size = 512
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,2])),
             'biases':tf.Variable(tf.random_normal([2]))}
    
    lstm_cell = rnn_cell.LSTMCell(rnn_size,reuse=None)
 
    outputs, states = rnn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def train_neural_network(learning_rate = 0.01, batch_size=1 ,hm_epochs=500):
    x_data, y_data = prepare_data("Wave_sliced")
#    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

    prediction = recurrent_neural_network()
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y[-1], logits = prediction))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    
    saver = tf.train.Saver();
    
    with tf.Session() as sess:
#        saver.restore(sess,"models/model.ckpt")
        sess.run(tf.global_variables_initializer())
                
#        train_dict = {x: x_data, y: y_data}
#        test_dict = {x: x_test, y: y_test}
    
        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            epoch_data = list(zip(x_data, y_data))
            np.random.shuffle(epoch_data)
            
            for x_sample,y_sample in epoch_data:                 
                epoch_x = x_sample.reshape((1,x_sample.shape[0],x_sample.shape[1]))                
                epoch_y = y_sample.reshape((1,y_sample.shape[0],y_sample.shape[1]))                                                
                _, c= sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c                
            average_loss = epoch_loss / len(epoch_data)
            print('Epoch', str(epoch), 'completed out of',hm_epochs,
                  'loss:', str(epoch_loss),
                  'average loss', str(average_loss))
        
        save_path = saver.save(sess, "models/model.ckpt")        
        print("Model saved in path: %s" % save_path)        
                
def get_accuracy(model,test_data):
    
    prediction = recurrent_neural_network()
    
    saver = tf.train.Saver();
    with tf.Session() as sess:
        saver.restore(sess,model)        
        x_test ,y_test = prepare_data(test_data)
        test_data = list(zip(x_test, y_test))
        
        hit = 0 # true positive
        miss = 0 # false negative
        false_alarm = 0 # false positive
        correct_rejection = 0  # true negative
        
        for x_sample,y_sample in test_data:
            
            y_prediction = sess.run(prediction, feed_dict={x:x_sample.reshape((1,x_sample.shape[0],x_sample.shape[1]))})
            y_prediction = np.argmax(y_prediction,1)
            y_labeled = np.argmax(y_sample,1)
            
            correct = np.equal(y_prediction,y_labeled)
            
            for i in range(len(correct)):
                if correct[i]:
                    if y_labeled[i] == 1:
                        hit = hit + 1
                    else: correct_rejection  = correct_rejection + 1
                else:
                    if y_labeled[i] == 1:
                        miss = miss + 1
                    else: false_alarm = false_alarm + 1
                    
        total_amount = hit + correct_rejection + miss + false_alarm            
        accuracy = float((hit + correct_rejection))/total_amount
        print("Accuracy: ",accuracy)
        
#        correct = np.equal(y_prediction,y_labeled)
##        correct = tf.equal(y_prediction,y_labeled)
#        print(correct)
#        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#        print(accuracy)
        
#        print('Accuracy:',accuracy.eval(train_dict))

def get_prediction(wav_file):
    
    sess = tf.Session()

    prediction = recurrent_neural_network()

    saver = tf.train.Saver()
    saver.restore(sess, "saved_models/First_model/model.ckpt")
    
    input_data = get_mfcc(wav_file)
    input_x = input_data.reshape((1,input_data.shape[0],input_data.shape[1]))    

    output_y = sess.run(prediction, feed_dict={x:input_x})
    output_y = np.argmax(output_y,1)
    print(output_y)   
    return output_y


def convert(y_output):
    
    annotations = []
    for i in range(len(y_output)):
        if y_output[i] == 1:
            j = i
            while y_output[j] == 1:
                j = j+1
            start = i*frame_step + frame_length/2
            end = (j - 1)*frame_step + frame_length/2
            annotations.append((start,end))
            i = j
    return annotations

train_neural_network()
#get_accuracy("saved_models/First_model/model.ckpt","Test_Data/")

#label_y = get_label("Test_Data/142_slices/142_part_1")
#label_y = np.argmax(label_y,1)
#
#test = get_prediction("Test_Data/142_slices/142_part_1")
#time = convert(test)
#time2 = convert(label_y)
#print(time)
#print(time2)
#print(len(label_y))

#print(label_y)
#y_prediction = get_prediction("Test_Data/142_slices/142_part_1")
#table = np.equal(label_y,y_prediction)
#print(table)

#T = textgrid.TextGrid();
#head,tail = os.path.split("Test_Data/142_slices/142_part_1")
#number = tail[0:3]
#T.read("TextGrids/"+number+"_wav.TextGrid")
#w_tier = T.getFirst("Vokale").intervals
## Generieren eines neuen Outputs/Labeling
#xmin1,xmax1,xmin2,xmax2,xmin3,xmax3, new_duration = getSegments("Test_Data/142_slices/142_part_1",w_tier)
#print(xmin1,xmax1,xmin2,xmax2,xmin3,xmax3,new_duration)










