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

from sklearn.metrics import roc_curve, auc

cmd_input = sys.argv[1]
model_type = cmd_input.lower()
model_name = model_type + "_"

def train_models():
    start = sys.argv[2]
    end = sys.argv[3]
    for i in range(int(start),int(end)):
        os.system("python3 model_trainer.py " +"model_sets/training_sets/"+model_type+"/"+model_name.title()+str(i))

def create_measurements():
    os.system("rm ROC_Values/"+model_type+"/"+"*_ROC")
    os.system("rm measurements/"+model_type+"*")
    for i in range(0,9):
        os.system("python model_trainer.py " + model_name+str(i))

def create_wrong_correct():
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
    for i in range(0,9):

#        roc_values = []
        
#        ROC_File = join("ROC_Values",model_type,model_name+str(i)+"_ROC")
#        f = open(ROC_File, "r+")
#        
#        roc = pickle.load(f)
        
        JSON_File = join("measurements",model_name+"measurement")
        h = open(JSON_File,"r")
        JSON = json.load(h)
        
#        for pair in roc:
#            roc_values.append(pair[2])
##            print(pair[2])
#            sorting_pair.append((pair[1],pair[0]))
#        
#        list.sort(sorting_pair)
#        for p in sorting_pair:
#            tpr.append(p[1])
#            fpr.append(p[0])
        tpr = JSON[model_type][i]["Recall"]
        fpr = JSON[model_type][i]["Fallout"]
        
        sorting_pair.append((fpr,tpr))
#        tprs.append(interp(mean_fpr, fpr,tpr))
#        tprs[-1][0] = 0.0
#        roc_auc = auc(fpr,tpr)
    
    
    list.sort(sorting_pair)
    print(sorting_pair)
    
    tprs = list([pair[1] for pair in sorting_pair])
    fprs = list([pair[0] for pair in sorting_pair])
    
    
    mean_tpr = np.mean(tprs, axis=0)

    roc_auc = auc(fprs,tprs,True)
    
    h = interp(mean_fpr,fprs,tprs)
    print(h)
#    aucs.append(roc_auc)
    
    
    tprs.append(1)
    fprs.append(1)
    tprs.insert(0,0)
    fprs.insert(0,0)    
    plt.plot(fprs,tprs,lw=1, alpha= 0.3 ,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
#    print(mean_tpr)
#    print(mean_fpr)
#    mean_tpr[-1] = 1.0
#    mean_auc = auc(mean_fpr, mean_tpr)
#    std_auc = np.std(aucs)
#    plt.plot(mean_fpr, mean_tpr, color='b',
#             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#             lw=2, alpha=.8)
#    
#    std_tpr = np.std(tprs, axis=0)
#    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                     label=r'$\pm$ 1 std. dev.')
#    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
            

train_models()
#create_measurements()
#plot_roc_curve()

