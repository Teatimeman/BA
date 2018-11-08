#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 23:07:40 2018

@author: teatimeman
"""

import json
import os
import pickle
import numpy as np
from os.path import isfile , isdir, join
import sys
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.interpolate import spline
from sklearn.metrics import roc_curve, auc



def train_models():
    cmd_input = sys.argv[1]
    model_type = cmd_input.lower()
    model_name = model_type + "_"
    n_mfcc = sys.argv[2]
    n_filt= sys.argv[3]
    start = sys.argv[4]
    end = sys.argv[5]
    for i in range(int(start),int(end)):
        os.system("python3 model_trainer.py " +n_mfcc+" " +n_filt+" model_sets/training_sets/"+model_type+"/"+model_name.title()+str(i))
train_models()
def create_measurements():
    cmd_input = sys.argv[1]
    model_type = cmd_input.lower()
    model_name = model_type + "_"
#    os.system("rm ROC_Values/"+model_type+"/"+"*_ROC")
#    os.system("rm measurements/"+model_type+"*")
    for i in range(0,9):
        os.system("python model_trainer.py " + model_name+str(i))
#create_measurements()
def get_wrong_correct():
    cmd_input = sys.argv[1]
    model_type = cmd_input.lower()
    model_name = model_type + "_"
    for i in range(0,9):
        model = model_name+str(i)+".ckpt"
        model_folder = model_name.title()+str(i)
        
        test_folder = join("model_sets/test_sets/", model_type, model_folder)
        file_liste = os.listdir(test_folder)
        model_path =  join("models",model_type,model_folder,model)
        
        for f in file_liste:
            filename = join(test_folder,f)
            os.system("python model_trainer.py " + filename +" "+ model_path)

def plot_roc_curve():
    
    tprs = []
    fprs = []
#    aucs = []
    mean_fpr = np.linspace(0,1,100)
    
    sorting_pair = []  
    
    curves = os.listdir("measurements")
    for curve in curves:    
        
        tprs = []
        fprs = []
    #    aucs = []
        for root, dic, files in os.walk("measurements/"+curve):
            for f in files:
                JSON_File = join(root,f)
                JSON = json.load(open(JSON_File,"r"))
                tpr = JSON["Measurements"][0]["Recall"]
                fpr = JSON["Measurements"][0]["Fallout"]
                sorting_pair.append((fpr,tpr))
            list.sort(sorting_pair)
            
            tprs = list([pair[1] for pair in sorting_pair])
            
            fprs = list([pair[0] for pair in sorting_pair])
            tprs.append(1)
            fprs.append(1)
            tprs.insert(0,0)
            fprs.insert(0,0)    
            power_smooth = spline(fprs,tprs,mean_fpr)
            roc_auc = auc(fprs,tprs,True)
            plt.plot(mean_fpr,power_smooth)
            plt.plot(fprs,tprs,lw=1, alpha= 1 ,label='ROC fold %s (AUC = %0.2f)' % (curve, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def create_mean_measures():
    types = os.listdir("measurements")
    for typ in types:
        
        accuracy = 0.
        balanced_accuracy = 0.
        mcc_accuracy = 0.
        
        hit =0.
        false_alarm  = 0.
        correct_rejection = 0.
        miss =0.
        
        recall =0.
        miss_rate =0.
        
        specificity =0.
        fallout =0.
        
        precision =0.
        npv =0.
        
        fcr = 0.
        presence =0.
        f_measure = 0.
        for root, dic, files in os.walk("measurements/"+typ):
            for f in files:
                JSON_File = join(root,f)
                JSON = json.load(open(JSON_File,"r"))
                accuracy += JSON["Measurements"][0]["Accuracy"]            
                balanced_accuracy += JSON["Measurements"][0]["Balanced_Accuracy"]
                mcc_accuracy += JSON["Measurements"][0]["MCC_Accuracy"]
                hit += JSON["Measurements"][0]["Hits"]
                false_alarm += JSON["Measurements"][0]["False_alarms"]
                correct_rejection += JSON["Measurements"][0]["Correct_rejections"]
                miss += JSON["Measurements"][0]["Misses"]
                recall += JSON["Measurements"][0]["Recall"]
                miss_rate += JSON["Measurements"][0]["Miss rate"]
                specificity += JSON["Measurements"][0]["Specificity"]
                fallout += JSON["Measurements"][0]["Fallout"]
                precision += JSON["Measurements"][0]["Precision"]
                npv += JSON["Measurements"][0]["npv"]
                fcr += JSON["Measurements"][0]["fcr"]
                presence += JSON["Measurements"][0]["Presence"]
                f_measure += JSON["Measurements"][0]["F-measure"]
            print(typ)
            print("accuracy: ",accuracy / 9.)
            print("balanced: ",balanced_accuracy / 9.)
            print("mcc: ",mcc_accuracy / 9.)
            print("hits: ",hit / 9.)
            print("false_alarm: ",false_alarm  / 9.)
            print("correct_rejection: ",correct_rejection / 9.)
            print("miss: ",miss / 9.)
            print("recall: ",recall / 9.)
            print("miss_rate: ",miss_rate / 9.)
            print("specificity: ",specificity / 9.)
            print("fallout: ",fallout / 9.)
            print("precision: ",precision / 9.)
            print("npv: ",npv  /9.)
            print("fcr: ",fcr / 9.)
            print("presence: ",presence / 9.)
            print("f_measure: ",f_measure / 9.)


#plot_roc_curve()

#create_mean_measures()
#x = (1,2,3,4,5,6)
#y =(5,6,7,8,9)
#print(9/3)
