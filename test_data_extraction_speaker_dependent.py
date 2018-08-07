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

folder = "Trainings_Data_Speaker_Dependent"

destination = "Test_Data_Speaker_Dependent"

substring055 = "_part_55"
substring061 = "_part_61"
substring067 = "_part_67"

paths = []

for root, directories, files in os.walk(folder):
    for filename in files:
        if substring055 in filename:
#            print(filename)
            paths.append(join(root,filename))
        if substring061 in filename:
            paths.append(join(root,filename))
        if substring067 in filename:
            paths.append(join(root,filename))
            
for path in paths:
    shutil.move(path,join(destination,os.path.basename(path)))

#signal_mfccs = [get_mfcc(s) for s in signals]
#signal_labels = [get_label(s) for s in signals] ## den output vector für alle signale erzeugen