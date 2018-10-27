#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 19:34:48 2018

@author: teatimeman
"""

import os
import shutil
from os import listdir
from os.path import isfile , isdir, join

folder = "Wave_sliced"

test_destination = "model_sets/test_sets/dependent"
training_destination = "model_sets/training_sets/dependent"

# Von jedem sprecher immer 3 Files, durchschnitt an Files eines Sprechers = 30
# 3 * 79 = 237 geschnitte Files zum Testen von insgesamt 2303 

dependent_1 = ("_1","_7","_13") 
dependent_2 = ("_19","_25","_31")
dependent_3 = ("_37","_43","_49")
dependent_4 = ("_55","_61","_67")
dependent_5 = ("_73","_79","_85")
dependent_6 = ("_91","_97","_103")
dependent_7 = ("_109","_115","_121")
dependent_8 = ("_127","_133","_139")
dependent_9 = ("_145","_151","_157")

data = [dependent_1, dependent_2, dependent_3, dependent_4, dependent_5 , dependent_6, 
        dependent_7, dependent_8, dependent_9]

i = 0

for d in data:
    i = i + 1
    target_folder = "Dependent_" + str(i)
    target_folder.endswith
    for root, directories, files in os.walk(folder):
        for filename in files:
            file_path= join(root,filename)
            if filename.endswith(d):
                target_destination = test_destination
            else:
                target_destination = training_destination     
            shutil.copyfile(file_path,join(target_destination,target_folder,filename))    
            

#signal_mfccs = [get_mfcc(s) for s in signals]
#signal_labels = [get_label(s) for s in signals] ## den output vector für alle signale erzeugen