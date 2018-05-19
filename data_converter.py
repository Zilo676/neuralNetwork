#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 20:37:18 2018

@author: zilo
"""

import pandas as pd
import numpy as np
import os
import sys

def get_file_list(dataset_path):
    if dataset_path[-1]!='/':
        dataset_path = dataset_path + '/'
    file_list = os.listdir(path=dataset_path)
    return file_list

def make_dataset_dir():
    new_dir='convertedDatasetFaults'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir
'''
в 0 max
в 1 min 
'''
def find_max_min(file_list,path):
    maxmin_storage=max_min_creator()
    for i in file_list:
        data = pd.read_csv(path+i,header=None,skiprows=1)
        for j in range(14):
            maxmin_storage[j][0]=data[j].max()
            maxmin_storage[j][1]=data[j].min()
    return maxmin_storage
        
def max_min_creator():
    maxmin_storage=np.zeros((1,14,2),dtype=float)
    for i in range(14):
        maxmin_storage[0][i][1]=sys.float_info.max
        maxmin_storage[0][i][0]=sys.float_info.min
    return maxmin_storage[0]  
  
def rescalling_convert(maxmin_storage,file_list,path):
    dirr=make_dataset_dir()
    for i in file_list:
        data = pd.read_csv(path+i,header=None,skiprows=1)
        rescalling_data=data
        print(i)
        for j in range(14):
            rescalling_data[j]=(data[j]-maxmin_storage[j][1])/(maxmin_storage[j][0]-maxmin_storage[j][1])
        rescalling_data.to_csv(dirr+'/'+i,header=None,index=False)
    print('Convert done')
path='faults/'    
file_list=get_file_list('dataset')
maxmin_storage=find_max_min(file_list,path)
rescalling_convert(maxmin_storage,file_list,path)
  
