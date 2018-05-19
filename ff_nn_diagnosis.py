#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 17:59:37 2018

@author: santit, zilo
"""

import nn_wrapper
from nn_factory import NN_factory
import data_source
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout 
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot



class FF_NN_factory(NN_factory):
    def __init__(self, input_shape, output_shape, labels, model_file_name, weight_file_name, auto_load = False):
        super().__init__(input_shape, output_shape, labels, model_file_name, weight_file_name, auto_load)
        if not auto_load:
            self._model = self.ff_model_generator()
            
    def ff_model_generator(self):
        print("Create new model")
        inputs = Input(shape=self._input_shape)
        drop_value = 0.5
        x = Dense(1400, activation = 'relu')(inputs)
        x = Dropout(drop_value)(x)
        x = Dense(1400, activation = 'relu')(x)
        #x = Dropout(drop_value)(x)
        #x = Dense(400, activation = 'relu')(x)
        #x = Dropout(drop_value)(x)
        #x = Dense(400, activation = 'relu')(x)
        #x = Dropout(drop_value)(x)
        #x = Dense(400, activation = 'relu')(x)
        x = Flatten()(x)
        predictions = Dense(self._output_shape, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=predictions)
        #SVG(model_to_dot(model).create(prog='dot', format='svg'))
        #plot_model(model,to_file='model.png')
        return model

class Multifault_diagnosis_NN(FF_NN_factory):
    def __init__(self):
        super().__init__(input_shape, 
             output_shape, 
             ["NORMAL", "FAULT_1","FAULT_2","FAULT_3","FAULT_4","FAULT_5","FAULT_6","FAULT_7","FAULT_8","FAULT_9",
              "FAULT_10","FAULT_11","FAULT_12","FAULT_13","FAULT_14"], 
             model_file_name + "_f1", 
             weight_file_name + "_f1", 
             auto_load)
        


def train_ff_model(nn_factory, data_source_dir, dataset_size, epochs=10):
    print("Prepare model")
    NN_model = nn_wrapper.Neural_network_wrapper(nn_factory)    
    NN_model.compile_model(loss='categorical_crossentropy')

    print("Prepare dataset")
    data_generator = data_source.data_source(data_source_dir, 100, 14, 15, overlap = 90)
    data_generator.prepare_random_sequence_batch(dataset_size, balancing_dataset=True)
    print("Training start")
    NN_model.train(data_generator, dataset_size, testing_ratio = 0.1, validation_split=0.3, epochs=epochs,view_result_diagramm=True)
    
    print("Saving model")
    nn_factory.save_model();
    print("Saving weight")
    nn_factory.save_weight();

model_file_name = "FF_diagnosis_model"
weight_file_name = "FF_diagnosis_weights"
input_shape = (100, 14) 
output_shape = 15
auto_load = False
n_epochs = 10
dataset_reduce_factor = 1

#data=data_source.data_source('dataset/',100,14,15,90,15)
#print("Dataset size " + str(data.test_data_count(balancing_dataset=True,max_diff=50000)))
#7208 - max_diff=50
#7955 - max_diff=100
#22547 - max_diff=1000
#566541 - max_diff=50000
#1374245 balancing_dataset=False

#train_ff_model(Multifault_diagnosis_NN(), './convertedDataset/', int(1374245/dataset_reduce_factor), epochs=n_epochs)
train_ff_model(Multifault_diagnosis_NN(), './convertedDataset/', int(7955/dataset_reduce_factor), epochs=n_epochs)


