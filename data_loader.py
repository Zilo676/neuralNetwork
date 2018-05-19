#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 21:50:13 2018

@author: santit, zilo
"""

import pandas as pd
import numpy as np
import os

class End_of_DB(Exception):
    def __init__(self, db_name):
        self.db_name = db_name
    def __str__(self):
        return "End of database: " + str(self.db_name)


class data_base:
    def __init__(self, name_db, batch_size, data_shift):
        """
        Инициализация класса data_base
        """
        self._name_db = name_db
        if self._name_db[-1]!='/':
            self._name_db = self._name_db + '/'
        self._file_list = os.listdir(path=self._name_db)
        self._cur_file_index = 0
        self._cur_index = 0
        self._batch_size = batch_size
        self._data_shift = data_shift
        self._current_data = self._load_from_file_by_index(self._cur_file_index)
            
    def _load_from_file(self, file_name):
        data = pd.read_csv(self._name_db + file_name, header=None,skiprows=1)
        return data.astype(np.float32)
    
    def _load_from_file_by_index(self, index):
        return self._load_from_file(self._file_list[index])

    def get_batch(self):
        if (self._cur_index + self._batch_size) >= self._current_data.shape[0]:
            if self._cur_file_index+1 >= len(self._file_list):
                raise End_of_DB(self._name_db)
            self._current_data = self._load_from_file_by_index(self._cur_file_index + 1)
            self._cur_file_index += 1
            self._cur_index = 0
            
        data = self._current_data[self._cur_index:self._cur_index + self._batch_size].as_matrix()
        self._cur_index += self._data_shift 
        return data
            