# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 17:00:26 2018

@author: Ethan
"""

import numpy as np

import json 
from collections import namedtuple

classes = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']

#x = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

class JSON_table():
    
    def __init__(self):
        json_data=open(r'.\data\examples.json').read()
        self.data = json.loads(json_data)

    def get_values(self, name):
        return self.data[name]
    
    def get_pitch(self,data):
        p = data['pitch']
        vec = np.zeros((1,128))
        vec[0,p] = 1
        return vec
    
    def get_velocity(self,data):
        v = data['velocity']
        vec = np.zeros((1,128))
        vec[0,v] = 1
        return vec
    
    def get_qualities(self,data):
        q = data['qualities']
        i = np.array(q)
        vec = np.zeros((1,10))
        vec[0,:] = i
        return vec
    
    def all_classes(self):
        return classes
        
    
    def get_info(self,name):
        json_ob = self.data[name]
        
        p = self.get_pitch(json_ob)
        v = self.get_velocity(json_ob)
        i = self.get_qualities(json_ob)
        
        return np.concatenate((p,v,i),axis=1)
        
def main():
    table = JSON_table()
    print(table.data["keyboard_acoustic_004-060-025"]['pitch'])
    
if __name__=="__main__":main()
        
        
    
