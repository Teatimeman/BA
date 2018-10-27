#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:13:02 2018

@author: teatimeman
"""

import os

for i in range(1,5):
    os.system("python3 model_trainer.py Independent_"+str(i))
#    os.wait()
    
