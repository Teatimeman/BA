#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 20:55:03 2018

@author: teatimeman
"""
import numpy as np
import tensorflow as tf
import model_trainer as mt

sess = tf.Session()

prediction = mt.recurrent_neural_network()

saver = tf.train.Saver()
saver.restore(sess, "saved_models/First_model/model.ckpt")

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y = graph.get_tensor_by_name("y:0")

input_data = mt.get_mfcc("base_line_signal")
input_x = input_data.reshape((1,input_data.shape[0],input_data.shape[1]))    

#label_y = get_label("Test_Data/142_slices/142_part_1")
#label_y = np.argmax(label_y,1)

output_y = sess.run(prediction, feed_dict={x:input_x})
output_y = np.argmax(output_y,1)

print(output_y)

#print(label_y)





