#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:32:19 2018

@author: teatimeman
"""

from pydub import AudioSegment

import textgrid
import re

import os
from os import listdir
from os.path import isfile , join

def slicer(folder):
    audioSignalName =  [signal for signal in listdir(folder) if isfile(join(folder,signal))]
    
    textgridFolder = folder + "TextGrids/"

    for signal in audioSignalName:
        newAudio = AudioSegment.from_wav(join(folder,signal))
         
        signalNumber = signal[0:3]
        signalFolder = "Wave_sliced/" + signalNumber + "_slices/"
        signalSlices = signalNumber + "_part_"
        
        if not os.path.exists(signalFolder):
            os.makedirs(signalFolder)
            
        sliceName = signalFolder + signalSlices
       
        textgridName =  signalNumber + "_wav.TextGrid"
        
        T = textgrid.TextGrid();
        T.read(textgridFolder + textgridName)
        
        T.write
        w_tier = T.getFirst("Vokale").intervals
        
        # Grenze zu den Wörtern mit einer Zahl ermitteln 
        i = 0
        j = len(w_tier)-1
        while i < len(w_tier):
            if w_tier[j].mark == "" or  re.match("\\d",w_tier[j].mark[-1]) == None:
                j -= 1
            i += 1
        j +=1
            
        # Erstellen von den Waves
        i = 1
        while i in range(j):
            
            d       = (w_tier[i+1].bounds()[1] - w_tier[i+1].bounds()[0])/2
            
            
            
            start   = (w_tier[i].bounds()[0] - d) * 1000
            end     = (w_tier[i+4].bounds()[1] + d) * 1000
            
            part    = newAudio[int(start):int(end)]                                    
            part.export(sliceName+str(i), format = "wav")                        
            
            
            
            i = i + 6

slicer("Daten_vowel_BA/RecordingVCP/new_data/")   
