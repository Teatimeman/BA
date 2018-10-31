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


# 8 sprecher zum testen mit jeweils ca 30 Files =  240 Files außer beim letzte



folder = "Wave_sliced"
female_folder = "new_data/female"
male_folder = "new_data/male"

test_destination = "model_sets/test_sets/independent"
training_destination = "model_sets/training_sets/independent"

test_list = []
training_list = []

female_files = os.listdir(female_folder)
list.sort(female_files)

male_files = os.listdir(male_folder)
list.sort(male_files)

for item in os.listdir(test_destination):
    test_list.append(join(test_destination,item))
list.sort(test_list)

for item in os.listdir(training_destination):
    training_list.append(join(training_destination,item))
list.sort(training_list)

def sets(gender_list,gender):        
        
        count = 0
        folder_number = 0
        for f in gender_list:
            if count == gender:
                count = 0
                folder_number = folder_number + 1 
                
            file_number = f[0:3]
            folder_name = file_number + "_slices"
            folder_path = join(folder,folder_name)
            
            test_folder = join(test_list[folder_number],  folder_name)            
            if not os.path.isdir(test_folder):
                shutil.copytree(folder_path,test_folder)
            
            for i in range(9):
                if not i == folder_number:
                    trainings_folder = join(training_list[i],folder_name)
                    if not os.path.isdir(trainings_folder):
                        shutil.copytree(folder_path,trainings_folder)
            
            count = count + 1   
        
        shutil.copytree(folder+"/"+"004_slices",test_destination+"/Independent_8/004_slices")
        
def get_data(test_set_of):
    start_female = test_set_of * 6
    end_female = test_set_of * 6 + 6 
    start_male = test_set_of * 2
    end_male = test_set_of * 2 + 2
    print("Female Data for Testing: " , female_files[start_female:end_female])
    print("Male Data for Testing: ", male_files[start_male:end_male])
    
sets(male_files,2)
sets(female_files,6)
#get_data(8)

#since the speaker gender distribution is not even results in a big varianz between models11