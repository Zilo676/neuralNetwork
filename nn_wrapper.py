#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 10:03:16 2018

@author: santit, zilo
"""

from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn import metrics
import keras
from nn_factory import NN_factory
import numpy as np
import matplotlib.pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class Neural_network_wrapper:
    VERSION = 2
    
    def __init__(self, 
                 model_factory):
        self._model_factory = model_factory
        self._model = self._model_factory.get_model()
        
    def compile_model(self, 
                      optimizer='adam', 
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'], 
                      loss_weights=None, 
                      sample_weight_mode=None, 
                      weighted_metrics=None, 
                      target_tensors=None):
        self._model.compile(optimizer, loss, metrics, loss_weights, 
                            sample_weight_mode, weighted_metrics, target_tensors)
    
    def train(self, 
              data_generator, 
              dataset_size, 
              testing_ratio = 0.2, 
              batch_size=None, 
              epochs=1, 
              verbose=1, 
              validation_split=0.0, 
              validation_data=None, 
              shuffle=True, 
              class_weight=None, 
              sample_weight=None, 
              initial_epoch=0, 
              steps_per_epoch=None, 
              validation_steps=None,
              view_result_diagramm=False):
        
        dataset_size_testing = int (testing_ratio * dataset_size)
        dataset_size_training = dataset_size - dataset_size_testing
        
        print('Preparing dataset for training')
        inputs = []
        outputs = []
        for _ in range(dataset_size_training):
            [i, o] = data_generator.get_rnd_data()
            inputs.append(i)
            outputs.append(o)
        print('Training')    
        history = LossHistory()
        checkpointer = ModelCheckpoint(filepath=self._model_factory.get_weight_file_name(), verbose=1, save_best_only=True)
        #TODO: Добавить в класс выбор типа логгера, в т.ч. TensorBoard
        inputs = np.squeeze(np.asarray(inputs))
        outputs = np.squeeze(np.asarray(outputs))
        self._model.fit(x=inputs, 
                        y=outputs, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        verbose=verbose, 
                        callbacks=[history, checkpointer], 
                        validation_split=validation_split, 
                        validation_data=validation_data, 
                        shuffle=shuffle, 
                        class_weight=class_weight, 
                        sample_weight=sample_weight, 
                        initial_epoch=initial_epoch, 
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps)
        
        print('Predicting')

        print('Preparing dataset for validating')
        testing_inputs = []
        testing_outputs = []
        for _ in range(dataset_size_testing):
            [i, o] = data_generator.get_rnd_data()
            testing_inputs.append(i)
            testing_outputs.append(o)
        testing_inputs = np.squeeze(np.asarray(testing_inputs))
        
        print('Predicting testing outputs')
        predicted_output = self.predict(testing_inputs)
        #to_categorial
        self._view_prediction_result(testing_outputs, predicted_output, 15, self._model_factory.get_labels(), view_diagram=view_result_diagramm)
        return history.losses
    
    def predict(self, 
                x, 
                batch_size=None, 
                verbose=0, 
                steps=None):
        return self._model.predict(x, batch_size, verbose, steps)
        
    def _view_prediction_result(self, 
                               true_value, 
                               predict_value, 
                               n_classes, 
                               labels,
                               view_diagram):
        #print(true_value)
        #print(predict_value)
        k=0
        for i in predict_value:
            k+=1
        print(k)
        predict_matrix=np.zeros((1,k))
        true_matrix=np.zeros((1,k))
        for i in range(k):
            true_matrix[0][i]=true_value[i].argmax()
            predict_matrix[0][i]=predict_value[i].argmax()
        #print(predict_matrix)
        #print(true_matrix)
        a=predict_matrix[0]
        b=true_matrix[0]
        #print(a)
        #print(b)
        c=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        confusion_matrix = metrics.confusion_matrix(b, a,c)
        print (confusion_matrix)
        normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
        print ()
        print ("Confusion matrix (normalised to % of total test data):")
        print (normalised_confusion_matrix)
        
        # Plot Results: 
        if view_diagram:
            width = 10
            height = 7
            plt.figure(figsize=(width, height))
            plt.imshow(
                normalised_confusion_matrix, 
                interpolation='nearest', 
                cmap=plt.cm.rainbow
            )
            plt.title("Confusion matrix \n(normalised to % of total test data)")
            plt.colorbar()
            tick_marks = np.arange(n_classes)
            plt.xticks(tick_marks, labels, rotation=90)
            plt.yticks(tick_marks, labels)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()