#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 03:50:00 2018

@author: teatimeman
"""



import sys
import os
from os.path import isfile , isdir, join

number = sys.argv[1]
folder = "models/independent/Independent_"+str(number)


for root, directories, files in os.walk(folder):
    for f in files:
        if f[:11] == "independent":
            src = f 
            dst = f[:11]+"_hamming"+f[11:]
#            print(dst)
            src = join(folder,src)
            dst = join(folder,dst)
#            src = folder + src
#            dst = folder + dst
#            print(dst)
            os.rename(src, dst)
        