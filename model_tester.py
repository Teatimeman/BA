#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 20:55:03 2018

@author: teatimeman
"""
import numpy as np
import tensorflow as tf
#import model_trainer as mt
from model_trainer import get_mfcc , recurrent_neural_network


tf.reset_default_graph()

# batch_Size, Sequence_length, n_mfcc
x = tf.placeholder(tf.float32, [None, None, 40])
# batch_Size, Sequence_length_labels
y = tf.placeholder(tf.float32, [None, None,2])


sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "models/model.ckpt")
prediction = recurrent_neural_network()

input_data = get_mfcc("Test_Data/142_slices/142_part_1")
input_x = input_data.reshape((1,input_data.shape[0],input_data.shape[1]))
    
output_y = sess.run(prediction, feed_dict={x:input_x})
        
print(output_y)






