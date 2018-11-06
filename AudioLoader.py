# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:01:54 2018

@author: Ethan
"""

import os
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import numpy as np
from scipy import io as sio
from random import shuffle

from STFT import STFT, ISTFT
from json_loader import JSON_table

class AudioLoader():
    
    def __init__(self):
        #self.filepath = "C:/Users/Ethan/Desktop/AudioGen/audio"
        self.filepath = r"C:\Users\Ethan\Desktop\NSynth\audio2"
        self._class_names = list(set([x.split('_')[0] for x in os.listdir("C:/Users/Ethan/Desktop/AudioGen/audio")]))
        self._class_names.sort()
        self.directories = os.listdir(self.filepath)
        self.filenames = self.allFiles()
        self.table = JSON_table()
        self.test_file = self.filenames[400]
        shuffle(self.filenames)
        self.index = 0
    
    def class_names(self):
       return self._class_names
        
    def allFiles(self):
        files = []
        files.extend([os.path.join(self.filepath, x) for x in os.listdir(self.filepath) if x.endswith(".wav")])
        #for d in self.directories:
            #files.extend([os.path.join(self.filepath, d, x) for x in os.listdir(os.path.join(self.filepath, d)) if x.endswith(".wav")])
        return files
        
    def getFile(self, filepath):
        x = np.zeros((1,25600))
        (fs,f) = read(os.path.join(filepath))
        n = np.size(f)
        if (n > 25600):
            n = 25600
        x[0,0:n] = f[0:n]
        u = np.mean(x[0,:])
        x = x-u
        rms = np.sqrt(np.mean(x**2))
        x = np.divide(x,rms)
        #x = np.divide(x,np.max(np.abs(x),1))
        return x
    
    def mu_encode(self, x):
        return np.sign(x)*np.log(1+255*np.abs(x))/np.log(1+255)
        
    def mu_decode(self, x):
        return np.sign(x)*((1.0/255)*(((1+255)**np.abs(x))-1))
            
    def writeFile(self, name, data):
        write(name, 16000, data)
        
    def get_json_info(self, file):
        name = self.filenames[self.index].split('\\')[-1].split('.')[0]
        info = self.table.get_info(name)
        return info
        
    def load_test(self):
        xbatch = np.zeros((1,25600))
        class_vec = np.zeros((1,11))
        x = self.getFile(self.test_file)
        xbatch[0,:] = x[0,:]
        self.index+=1
        cond = self.get_json_info(self.test_file)
        c = self.test_file.split('_')[0].split('\\')[-1]
        class_vec[0,self._class_names.index(c)] = 1
        return xbatch,class_vec,cond
               
    def load_next(self, n=10):
        xbatch = np.zeros((n,25600))
        class_vec = np.zeros((n,11))
        conditions = np.zeros((n,266))
        while 1:
            for i in range(n):
                x = self.getFile(self.filenames[self.index])
                xbatch[i,:] = x
                cond = self.get_json_info(self.filenames[self.index])
                conditions[i,:] = cond
                c = self.filenames[self.index].split('\\')[-1].split('_')[0]
                class_vec[i,self._class_names.index(c)] = 1
                self.index+=1
                if self.index > len(self.filenames) -1:
                    self.index = 0
                    shuffle(self.filenames)   
            yield xbatch, class_vec, conditions

    def load_next_recon(self):
        xbatch = np.zeros((10,1,25600))
        ybatch = np.zeros((10,1,25600))
        while 1:
            for i in range(10):
                x = self.getFile(self.filenames[self.index])
                x = np.divide(x,max(x))
                self.index += 1
                y = ISTFT(STFT(x))
                xbatch[i,:,0] = x
                ybatch[i,:,0] = y[:,0]
            yield xbatch, ybatch
            
    def load_next_spec(self):
        xbatch = np.zeros((1,512,512,2))
        while 1:
            for i in range(1):
                x = self.getFile(self.filenames[self.index])
                self.index += 1
                spec = STFT(x)
                xbatch[i,:,:,:] = spec[0:512,:,:] 
            yield xbatch, xbatch
            
            
    def select_by_sample(self, name, n=4):
        xbatch = np.zeros((n,25600))
        instruments = np.zeros((n,len(self._class_names)))
        conditions = np.zeros((n,266))
        samples = [x for x in self.filenames if name in x]
        shuffle(samples)
        for i in range(n):
            x = self.getFile(samples[i])
            xbatch[i,:] = x
            cond = self.get_json_info(samples[i])
            conditions[i,:] = cond
            c = samples[i].split('\\')[-1].split('_')[0]
            instruments[i,self._class_names.index(c)] = 1
        yield xbatch, instruments, conditions

    def save(self,data):
       sio.savemat('test.mat',data)

if __name__ == "__main__": 
    A = AudioLoader()
    print(A.table.get_info(name=A.filenames[0].split('\\')[-1].split('.')[0]))
    #print(next(A.select_by_sample('bass')))
    