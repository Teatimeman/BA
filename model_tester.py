#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 20:55:03 2018

@author: teatimeman
"""
import numpy as np
import tensorflow as tf

from model_trainer import get_mfcc

saver = tf.train.Saver()
with tf.session as sess:
    
    saver.restore(sess, "models/model.ckpt")
    
    y_prediction = [];
    input_data = get_mfcc("Test_Data/142_slices/142_part_1")
    input_x = input_data.reshape((1,input_data.shape[0],input_data.shape[1]))
    
    sess.run(y_prediction, feed_dict={x:input_x})
        







