
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 19:13:28 2018

@author: santit, zilo
"""
from keras.models import Model, model_from_json

class NN_factory():
    def __init__(self, input_shape, output_shape, labels,  model_file_name, weight_file_name, auto_load = False):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._labels = labels
        self._model_file_name = model_file_name
        self._weight_file_name = weight_file_name
        self._model = None
        if auto_load:
            self.load_model_from_file()
            self.load_weight_from_file()
        
    def get_model(self):
        return self._model
    
    def get_labels(self):
        return self._labels
    
    def get_model_file_name(self):
        return self._model_file_name
    
    def get_weight_file_name(self):
        return self._weight_file_name
    
    def save_model(self):
        model_json = self._model.to_json()
        json_file = open(self._model_file_name, "w")
        json_file.write(model_json)
        json_file.close()
    
    def save_weight(self):
        self._model.save_weights(self._weight_file_name)    
    
    def load_model_from_file(self, file_name = None):
        if file_name is None:
            file_name = self._model_file_name
        print("Loading model from " + file_name)
        json_model_file = open(file_name, "r")
        loaded_model_json = json_model_file.read()
        json_model_file.close()
        self._model = model_from_json(loaded_model_json)
        
    def load_weight_from_file(self, file_name = None):
        if file_name is None:
            file_name = self._weight_file_name
        print("Loading weights from " + file_name)
        self._model.load_weights(file_name)