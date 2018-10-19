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



from python_speech_features import mfcc
from python_speech_features import fbank


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
    lstm_cell = rnn_cell.LSTMCell(rnn_size,reuse=None)
    rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = 0.7)
    outputs, states = rnn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
   
    return output

def train_neural_network(trainings_folder,learning_rate = 0.01, batch_size=1 ,hm_epochs=101):

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
            
            for x_sample,y_sample in epoch_data:                 
                epoch_x = x_sample.reshape((1,x_sample.shape[0],x_sample.shape[1]))                
                epoch_y = y_sample.reshape((1,y_sample.shape[0],y_sample.shape[1]))                                                
                _, c= sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c                
            average_loss = epoch_loss / len(epoch_data)
            print('Epoch', str(epoch), 'completed out of',hm_epochs,
                  'loss:', str(epoch_loss),
                  'average loss', str(average_loss))        
            
#            save_path = saver.save(sess, "models/"+trainings_folder+"/model_step_"+str(epoch)+"_.ckpt")
            
#            if epoch == 100:
#                save_path = saver.save(sess, "models/"+trainings_folder+"/100er_model/model_step_"+str(epoch)+"_.ckpt")
#                print("Model saved in path: %s" % save_path)
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
        
        for x_sample,y_sample in test_data:
            
            y_prediction = sess.run(prediction, feed_dict={x:x_sample.reshape((1,x_sample.shape[0],x_sample.shape[1]))})
            y_prediction = np.argmax(y_prediction,1)
            y_labeled = np.argmax(y_sample,1)
            correct = np.equal(y_prediction,y_labeled)
            thisCorrectAmount = 0
            for i in range(len(correct)):
                if correct[i]:
                    thisCorrectAmount = thisCorrectAmount + 1
                    if y_labeled[i] == 1:
                        hit = hit + 1
                    else: correct_rejection  = correct_rejection + 1
                else:
                    if y_labeled[i] == 1:
                        miss = miss + 1
                    else: false_alarm = false_alarm + 1
#            thisTotalAmount = len(correct)
#            thisAccuracy = float(thisCorrectAmount)/thisTotalAmount
#            print(thisAccuracy)
        total_amount = hit + correct_rejection + miss + false_alarm            
        accuracy = float((hit + correct_rejection))/total_amount
        presence =  float(hit + miss) / total_amount
        precision =  float(hit) / (hit + false_alarm)
        recall = float(hit) / (hit + miss)
        f_measure = 2 *(precision*recall) / (precision+recall)
        print("Name: ",test_folder)
        print("Hits: " , hit)
        print("False_alarms: ", false_alarm)
        print("Correct_rejections: ", correct_rejection)
        print("Misses: ", miss)
        print("Accuracy: ",accuracy)
        print("Presence: ",presence)
        print("Precision: ",precision)
        print("Recall: ", recall)
        print("F-measure:" , f_measure)
        print("total amount: ", total_amount)
        print("hits: ", hit)
        print("correct_rejections: ", correct_rejection)
        print("misses: ", miss)
        print("false_alarm: ",false_alarm)
        tf.get_default_session().close()
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
    output_y = np.argmax(output_y,1)
    
    print(output_y)   
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

def logscale_spec(spec, sr=16000, factor=20.):
    timebins, freqbins = np.shape(spec)
    
    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    scale = [int(i) for i in scale]
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs

def plot(signal,binsize= 2**10,plotpath = None,colormap = "jet"):
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - window_length)) / window_step))
    
    pad_signal_length = num_frames * window_step + window_length
    z = np.zeros(int(pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, window_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * window_step, window_step), (window_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    frames *= np.hamming(window_length)
    
    NFFT = 512    
    nfilt = 40
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sr)
    
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    
    filter_banks = 20 * np.log10(filter_banks)  # dB
    
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

    ims = filter_banks
    
    sshow, freq = logscale_spec(ims, factor=1.0, sr=sr)

    
    timebins, freqbins = np.shape(ims)
        
    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar(format = "%+2.0f dB")

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(signal)/timebins)+(0.5*binsize))/sr])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    
    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()
    plt.clf()
    
    
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


#print("Argument List:", str(sys.argv))
#print(sys.argv[1])
train_neural_network("model_sets/training_sets/dependent/" + str(sys.argv[1]))
        
#fs = 16000
#duration = 3
#myrecording = sd.rec(duration * fs , samplerate = fs , channels=2 , dtype = 'float64')
#print("recording...")
#sd.wait()
#sd.play(myrecording, fs)
#sd.wait()
#print("finish")

#for root, directories, files in os.walk("Test_Data_SpeakerDependent"):
#        for directory in directories:
#            get_accuracy("saved_models/First_model/model.ckpt",join(root,directory))

#get_accuracy("models/Trainings_Data_Speaker_Dependent/100er_model/model_step_100_.ckpt","Test_Data_Speaker_Dependent")

#train_neural_network("Trainings_Data_Speaker_Independent")
#get_accuracy("models/Trainings_Data_Speaker_Independent/100er_model/model_step_100_.ckpt","Test_Data_Speaker_Independent")

#get_accuracy("saved_models/First_model/model.ckpt","Test_Data/")
#get_accuracy("saved_models/First_model/model.ckpt","Test_Data_Speaker_Dependent/")
#get_accuracy("saved_models/100er_models/model_step_100_.ckpt","Test_Data/")
#get_accuracy("saved_models/100er_models/model_step_100_.ckpt","Test_Data_Speaker_Dependent/")
#get_baseLine_accuracy("Test_Data/")

#label_y = get_label("Test_Data/142_slices/142_part_1")
#label_y = np.argmax(label_y,1)
#test = get_prediction("Test_Data/142_slices/142_part_1","saved_models/First_model/model.ckpt")
#print(label_y)
#test = get_prediction("Test_Data/142_slices/142_part_1","saved_models/100er_models/model_step_100_.ckpt")
#time = convert(test)
#time2 = convert(label_y)
#print(time)
#print(time2)
#print(len(label_y))

#test = get_prediction("Test_Data/beat.wav","saved_models/First_model/model.ckpt")
#test = get_prediction("Test_Data/had.wav","saved_models/First_model/model.ckpt")
#test = get_prediction(myrecording,"saved_models/First_model/model.ckpt")
#time = convert(test)
#print(time)

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

#Notiz:
#10% auslassen rest training und geschlecht balancieren
#10% (6) davon frauen und 10% (2) männer 
#62 Frauen und 18 Männer insgesamt 
# 1 textgrid ohne inhalt darum eine Trainingsdatei fällt weg (NR 116 männlich)

#speaker dependent train score / test score
#speaker independent test scores 
#-> dropout erhöhen wenn delta zu hoch
#spectrogramm mit fbank grenzen mit ax vlines

# Verteidigung: 20 minuten vortrag  + 10 min fragen 
#fragen ob nach oder vor abgabe der thesis


