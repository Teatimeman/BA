#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:56:10 2017

@author: teatimeman
"""
#import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa
#import librosa.displays
import re
import json

import pickle

from python_speech_features import mfcc
from python_speech_features import fbank


from scipy import interp
from sklearn.metrics import roc_curve, auc

import sys

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


frame_length = 0.025
frame_step = 0.01
n_mfcc = 40
sr = 16000
window_length = frame_length * sr
window_length = int(round(window_length))
window_step = frame_step *sr
window_step = int(round(window_step))

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

def get_fbank(audioSignal):
    signal, samplerate = librosa.load(audioSignal,sr = sr)
    fbank_features = fbank(signal,samplerate,winlen=frame_length,winstep=frame_step,nfilt=40)
    return fbank_features[0]
    
def get_mfcc(audioSignal):
    signal, samplerate  = librosa.load(audioSignal,sr = sr)
    mfcc_signal = mfcc(signal,samplerate,winlen=frame_length,winstep=frame_step,nfilt=40,numcep=40) #,winfunc= np.hamming
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
 
    #words_raw = [[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0], ., ,[0.0,1.0],[0.0,1.0],[0.0,1.0],...      ] 
#    print(words_raw)
    words_label = np.asarray(words_raw, dtype = float)
#    print(words_label)
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
    
# batch_Size, Sequence_length, n_mfcc
x = tf.placeholder(tf.float32, [None, None, n_mfcc])
# batch_Size, Sequence_length_labels
y = tf.placeholder(tf.float32, [None, None,2])
    
def recurrent_neural_network():
#     x = tf.reshape(x, [-1, chunk_size])
    # x = tf.split(x, n_chunks)
    rnn_size = 512

    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,2])),
             'biases':tf.Variable(tf.random_normal([2]))}
    lstm_cell = rnn_cell.LSTMCell(rnn_size,reuse = False)
#    lstm_cell = rnn_cell.LSTMCell(rnn_size,reuse=True)
    rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = 0.7)
    outputs, states = rnn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
   
    return output

def train_neural_network(trainings_folder, batch_size=1 ,learning_rate = 0.01,hm_epochs=101):

    with tf.device("/GPU:0"):
        x_data, y_data = prepare_data(trainings_folder)
    #    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    
        prediction = recurrent_neural_network()
        
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y[-1], logits = prediction))
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
        
    with tf.Session() as sess:
#        saver.restore(sess,"models/model.ckpt")
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
#        train_dict = {x: x_data, y: y_data}
#        test_dict = {x: x_test, y: y_test}
        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            epoch_data = list(zip(x_data, y_data))
            np.random.shuffle(epoch_data)
            
            #stochastic gradient descent
#            for x_sample,y_sample in epoch_data:                 
#                epoch_x = x_sample.reshape((1,x_sample.shape[0],x_sample.shape[1]))                
#                epoch_y = y_sample.reshape((1,y_sample.shape[0],y_sample.shape[1]))                                                
#                _, sample_loss= sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
#                epoch_loss += sample_loss     

            # MiniBatch/Batch Gradient Descent 
            for start,end in zip(range(0,len(epoch_data), batch_size),range(batch_size,len(epoch_data)+1,batch_size)):

                epoch_x = [pair[0] for pair in epoch_data[start:end]] # x values for batch b with b = epoch_data[start:end]
                epoch_y = [pair[1] for pair in epoch_data[start:end]] # y values for batch f 

                _,batch_loss =sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += batch_loss
                
            if len(epoch_data) % batch_size !=0:  
                missing_samples = batch_size - len(epoch_data)%batch_size #  1 < missing_samples  < batch_size
                missing_start = len(epoch_data) - missing_samples
                
                last_batch = epoch_data[0:missing_samples] + epoch_data[missing_start:len(epoch_data)]
                
                last_batch_x = [pair[0] for pair in last_batch]
                
                last_batch_y = [pair[1] for pair in last_batch]
                
            _,batch_loss =sess.run([optimizer, cost], feed_dict={x: last_batch_x, y: last_batch_y})
            epoch_loss += batch_loss
            
            average_loss = epoch_loss / len(epoch_data)
            print('Epoch', str(epoch), 'completed out of',hm_epochs,
                  'loss:', str(epoch_loss),
                  'average loss', str(average_loss))   
            
        #saving model    
        model_kind = os.path.basename(os.path.dirname(trainings_folder))
        model_number = os.path.basename(trainings_folder)
        model_name = model_number.lower()
        save_path = saver.save(sess,"models/"+model_kind + "/" + model_number+ "/" + model_name + ".ckpt")
        print("Model saved in path: %s" % save_path)
                    
def get_accuracy(model,test_folder):
    

    prediction = recurrent_neural_network()
    
    saver = tf.train.Saver();
    with tf.Session() as sess:
        saver.restore(sess,model)        
        x_test ,y_test = prepare_data(test_folder)
        test_data = list(zip(x_test, y_test))
        
        hit = 0 # true positive
        miss = 0 # false negative
        false_alarm = 0 # false positive
        correct_rejection = 0  # true negative
        
        y_probabilities = []
        y_predictions = []
        
        y_labels = []
        y_samples = []
        
        tpr = []
        fpr = []
        roc_values = []

        sample_number= 0
        
        wrong_samples = 0
        correct_samples = 0
        
        
        for x_sample,y_sample in test_data:
            
            y_prediction = sess.run(prediction, feed_dict={x:x_sample.reshape((1,x_sample.shape[0],x_sample.shape[1]))})
            y_probabilities.append(y_prediction)
            
            y_prediction = np.argmax(y_prediction,1)
            y_predictions.append(y_prediction)
            
            y_labeled = np.argmax(y_sample,1)
            y_labels.append(y_labeled)
            
            y_samples.append(y_sample)
            
            correct = np.equal(y_prediction,y_labeled)
            
            thisCorrectAmount = 0
            this_hit = 0
            this_miss = 0
            this_false_alarm = 0
            this_correct_rejection = 0
            
            for i in range(len(correct)):
                
                if correct[i]:
                    thisCorrectAmount = thisCorrectAmount + 1
                    if y_labeled[i] == 1:
                        hit = hit +1
                        this_hit = this_hit + 1
                    else: 
                        this_correct_rejection  = this_correct_rejection + 1
                        correct_rejection = correct_rejection + 1 
                else:
                    if y_labeled[i] == 1:
                        this_miss = this_miss + 1
                        miss = miss +1
                    else: 
                        this_false_alarm  = this_false_alarm + 1
                        false_alarm = false_alarm + 1
#            thisTotalAmount = len(correct)
#            thisAccuracy = float(thisCorrectAmount)/thisTotalAmount
#            print(thisAccuracy)
            
            if  not (this_hit + this_miss) == 0:           
                this_recall = float(this_hit) / (this_hit + this_miss)
            else:
                wrong_samples = wrong_samples + 1
                this_recall = 1.0
                
            if not (this_false_alarm +this_miss) == 0:
                this_fallout = float(this_false_alarm) / (this_false_alarm + this_miss)
            else:
                this_fallout = 0.0
                correct_samples = correct_samples + 1
              
                
                
      
            tpr.append(this_recall) 
            fpr.append(this_fallout)

            roc_values.append((sample_number,this_recall,this_fallout))
            sample_number = sample_number + 1
        
        
        total_amount = hit + correct_rejection + miss + false_alarm            
        
        recall = float(hit) / (hit + miss)                          #true positive rate or sensitivity
        miss_rate = float(miss) / (hit + miss)    
    
        specificity = float(correct_rejection) / (correct_rejection + false_alarm)
        fallout = float(false_alarm) / (correct_rejection + false_alarm)
        
        precision =  float(hit) / (hit + false_alarm)
        npv = float(correct_rejection) / (miss + correct_rejection)
        
        accuracy = float((hit + correct_rejection))/total_amount
        fcr = float((false_alarm + miss))/total_amount
        
        presence =  float(hit + miss) / total_amount
        f_measure = 2 *(precision*recall) / (precision+recall)
        
        model_name = os.path.basename(os.path.normpath(model))
        
        print("Name: ",model_name)

        print("Hits (tp): " , hit)
        print("False_alarms (fp): ", false_alarm)
        print("Correct_rejections (tn): ", correct_rejection)
        print("Misses (fn): ", miss)
        
        print("Recall (tpr/sensitivity): ", recall)
        print("Miss rate (fnr): " , miss_rate)
        
        print("specificity (tnr/correct rejection rate): ",specificity)
        print("Fallout (fpr): ", fallout)
        
        print("Precision: ",precision)        
        print("Negative predictive value (npv): ", npv)
        
        
        print("Accuracy: ",accuracy)
        print("False classification rate: ", fcr)
        
        print("Presence: ",presence) # prozentualer anteil von 1 in
        print("F-measure:" , f_measure)       

        print("Total amount: ", total_amount)        

        print("Sample number: ", sample_number)
        print("Correct samples: ", correct_samples)
        print("Wrong samples: " , wrong_samples)
        
        return hit,false_alarm,correct_rejection,miss,recall, miss_rate,specificity,fallout ,precision,npv, accuracy, fcr,presence,f_measure, total_amount, y_probabilities, y_predictions , y_labeled, y_samples, tpr,fpr, roc_values
        
#        correct = np.equal(y_prediction,y_labeled)
##        correct = tf.equal(y_prediction,y_labeled)
#        print(correct)
#        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#        print(accuracy)
        
#        print('Accuracy:',accuracy.eval(train_dict))


# Vergleich der test_Daten mit einem Baseline Signal in dem Fall ein y_output 
# der nur 0 hat -> die zahl die daraus folgt gibt aufschluss darüber das: 
# mehr als die hälfte der Zahlen 0 sind, wenn die Accuracy eines models größer 
# als die Baseline Accuracy ist bedeutet es das das model in die richtige 
# richtung geht 
def get_baseLine_accuracy(test_data):
    x_test ,y_test = prepare_data(test_data)
    correct_rejection = 0
    miss = 0
    for y_sample in y_test:
        y_labeled = np.argmax(y_sample,1)        
        baseLine = np.zeros(len(y_labeled))
        correct = np.equal(baseLine, y_labeled)
    
        for i in range(len(correct)):
            if correct[i]:  
                correct_rejection = correct_rejection + 1        
            else:
                miss = miss + 1
    total_amount = miss + correct_rejection
    Accuracy = float(correct_rejection) / total_amount
    print("Baseline Accuracy: ", Accuracy)

# Gibt ein 1-dimensionales array wieder mit 0 oder 1 
def get_prediction(wav_file,model):
    
    sess = tf.Session()

    prediction = recurrent_neural_network()

    saver = tf.train.Saver()
    saver.restore(sess, model)
    
    input_data = get_mfcc(wav_file)
    input_x = input_data.reshape((1,input_data.shape[0],input_data.shape[1]))    

    output_y = sess.run(prediction, feed_dict={x:input_x})
#    print(output_y)
    output_y = np.argmax(output_y,1)
#    print(output_y)
    
    return output_y

# konvertiert einen y_output in zeit maße um
def convert(y_output):
    annotations = []
    i = 0
    while i < len(y_output):
        if y_output[i] == 1:
            j = i
            while y_output[j] == 1:
                j = j+1
            start = i*frame_step + frame_length/2
            end = (j - 1)*frame_step + frame_length/2
            annotations.append((start,end))
            i = j - 1
        i = i +1
    return annotations

    
def textGrid():    
    test = get_prediction("Test_Data/beat.wav","models/Trainings_Data_Speaker_Independent/100er_model/model_step_100_.ckpt")
    time = convert(test)
    print(len(test))
    print(time)
    print(len(time))
    print(time[1][1])
    T = textgrid.TextGrid(minTime = 0, maxTime = 10)
    Vokale = textgrid.IntervalTier("Vokale", minTime = 0, maxTime = 10)
    Interval1 = textgrid.Interval(minTime = time[0][0],maxTime = time[0][1], mark = "Wort1")
    #Vokale.add()
    Vokale.addInterval(Interval1)
    T.append(Vokale)
    T.write("Moses")

#plot dependent or independent roc curve
#def plot_roc(models):
    

def get_measurements(model_name):
    
    model_type = model_name[0:-2]
    model_Folder = model_name.title()

    model = join("models", model_type, model_Folder,  model_name+".ckpt")
    test_folder = join("model_sets/test_sets/", model_type, model_Folder)
    
    hit,false_alarm,correct_rejection,miss,recall, miss_rate, specificity,fallout ,precision,npv, accuracy, fcr,presence,f_measure, total_amount,y_probabilities,y_predictions, y_labeled, y_samples, tpr,fpr,roc_values= get_accuracy(model,test_folder)
    
    
#    ROC_File = join("ROC_Values", model_type, model_name+"_ROC")
##    y_all = [y_probabilities,y_predictions,  y_labeled, y_samples]
#    
#    roc  = [(tpr[i],fpr[i],roc_values[i]) for i in range(len(tpr))]
#    
#    with open(ROC_File,"wb+") as y:
#        pickle.dump(roc,y)
##    
    
    JSON_file = join("measurements", model_type+"_measurement")
    
    
    if isfile(JSON_file): 
        f = open(JSON_file)
        data = json.load(f)
    else:
        data = {}
        data[model_type] = []
           
    data[model_type].append({
            "Name": model_name,
            "Hits": hit,
            "False_alarms": false_alarm,
            "Correct_rejections": correct_rejection,
            "Misses":  miss,
            "Recall": recall,
            "Miss rate": miss_rate,
            "Specificity" : specificity,
            "Fallout":fallout,
            "Precision": precision,
            "npv":npv,
            "Accuracy": accuracy,
            "fcr":fcr,
            "Presence": presence,
            "F-measure": f_measure,
            "Total amount": total_amount,
            })       
    outfile = open(JSON_file, "wb+")
    json.dump(data,outfile)
        

def determine_correct_wrong(wav_path,model):
    label_y = get_label(wav_path)
    label_y = np.argmax(label_y,1)
    
    test = get_prediction(wav_path,model)
    correct = np.equal(label_y,test)
    
    model_name = os.path.basename(model)[0:-5]
    model_type = os.path.basename(model)[0:-7]
    
    wrong_correct_file = join ("wrong_correct",model_type, model_name +"_wc")
    output =  open(wrong_correct_file, "wb+")
    
    
    if isfile(wrong_correct_file): 
        f = open(wrong_correct_file,"rb+")
        data = json.load(f)
    else:
        data = {}
        data["wrong"] = []
        data["correct"] =[]  

    if not any(c == False for c in correct):
        data["correct"].append(wav_path)        
                
    elif not any(c == True for c in correct):
        data["wrong"].append(wav_path)     
        
    json.dump(data,output)
    
#    time = convert(test)
#    time2 = convert(label_y)
#    print(time)
#    print(time2)
#    print(len(label_y))
#    print(correct)

model_training_path = sys.argv[1]
batch_size = 32
train_neural_network(model_training_path,batch_size)

#model_name = sys.argv[1]
#get_measurements(model_name)

#wav_path = sys.argv[1]
#model_path = sys.argv[2]
#determine_100_correct(wave_path,sys.argv[2])    

#model_path = sys.argv[1]
#test_folder =sys.argv[2]
#get_accuracy(model, testfolder)

#Notiz:
#10% auslassen rest training und geschlecht balancieren
#10% (6) davon frauen und 10% (2) männer 
#62 Frauen und 18 Männer insgesamt 
# 1 textgrid ohne inhalt darum eine Trainingsdatei fällt weg (NR 116 männlich)

#speaker dependent train score / test score
#speaker independent test scores 
#-> dropout erhöhen wenn delta zu hoch
#spectrogramm mit fbank grenzen mit ax vlines