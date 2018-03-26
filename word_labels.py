#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:47:55 2018

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


T = textgrid.TextGrid();
T.read("new_data/004_wav.TextGrid")

w_tier = T.getFirst("Vokale").intervals

time_mark = 0.025/2 + 0.01*np.arange(0, mfcc_raw.shape[0])
time_mark = time_mark.astype('float32')

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

print("shape: "+ words_label.shape)


file = open("004_words_label.txt","w") 
 
file.write(words_label) 
 
file.close() 
