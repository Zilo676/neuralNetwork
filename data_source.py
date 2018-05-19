#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 18:49:23 2018

@author: santit, zilo
"""

import data_loader
import random
import numpy as np

class data_source():
    def __init__(self, name_db, batch_size, input_count, 
                 output_count = 15, overlap = 0, class_count = 15):
        self._loader = data_loader.data_base(name_db, batch_size, batch_size - overlap)
        self._input_count = input_count
        self._output_count = output_count
        self._batch_size = batch_size
        self._rnd_db = None
        self._class_count = class_count
        
    def test_data_count(self, balancing_dataset=False, max_diff = 1000):
        count = 0
        try:
            if balancing_dataset:
                counters = np.zeros(self._class_count)
                while True:
                    d = self.get_next_data()
                    if ((counters.argmax() != d[1].argmax() and counters[int(d[1].argmax())] - np.amax(counters) <= max_diff) 
                            or (counters.argmax() == d[1].argmax() and counters[int(d[1].argmax())] - np.amin(counters) <= max_diff) 
                            or np.amax(counters) == 0):
                        counters[int(d[1].argmax())] += 1
                        count += 1
            else:
                while (1):
                    self.get_next_data()
                    count += 1
        except Exception:
            return count
            
    def get_next_data(self):
       # try:
       data = self._loader.get_batch()
       inputs = data[:, :self._input_count]
       output = data[-int(self._batch_size), -self._output_count:]
       return [inputs, output ]
        #except Exception:
           # print('Dataset ended')
            #inputs=
    
    def prepare_random_sequence_batch(self, sequence_size, balancing_dataset=False, max_diff = 100):
        '''
        Подготовить случайную выборку размеченных данных
        Внимание. Балансировра работает только при условии 1 выхода на любое 
        количество классов
        '''
        self._rnd_db = []
        
        if balancing_dataset:
            counters = np.zeros(self._class_count)
            s = 0
            while s < sequence_size:
                d = self.get_next_data()
                if ((counters.argmax() != d[1].argmax() and counters[int(d[1].argmax())] - np.amax(counters) <= max_diff) 
                        or (counters.argmax() == d[1].argmax() and counters[int(d[1].argmax())] - np.amin(counters) <= max_diff) 
                        or np.amax(counters) == 0):
                    counters[int(d[1].argmax())] += 1
                    self._rnd_db.append(d)
                    s += 1
        else:
            for _ in range(sequence_size):
                self._rnd_db.append(self.get_next_data())
        
        random.shuffle(self._rnd_db)
        
        
        
    def get_rnd_data(self):
        if self._rnd_db is None:
            return None
        if len(self._rnd_db) == 0:
            return None
        return self._rnd_db.pop()
'''        
db = data_source('../Datasets/f1', 100, 4, 1, overlap = 90)

for _ in range(100000):
    [i, o] = db.get_next_batch()
    print(o)
'''