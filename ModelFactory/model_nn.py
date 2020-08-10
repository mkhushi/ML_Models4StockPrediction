# -*- coding: utf-8 -*-
'''
Created on Sun Oct  7 13:51:52 2018

@author: Thomas Brown
'''
## Importing the necessary packages

import numpy as np
import matplotlib.pyplot as plt

from keras import layers
from keras.models import Sequential

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from ModelFactory.model import Model, Configs

class NN(Model): # pylint: disable=too-many-instance-attributes
    '''Artificial neural network model'''

    def __init__(self, config, experiment):
        super(NN, self).__init__(config, experiment)
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_list = None
        self.y_test_list = None

    def set_params(self):
        '''prepares the hypermarameters to be used by the model'''

        if self.config is Configs.Config1:
            self.model_params = {
                'optimizer': 'adadelta',
                'epochs': 200,
                'learning_rate' : 0.2,
                'batch_size': 16,
                'dropout' : 0.5
                }
        elif self.config is Configs.Config2:
            self.model_params = {
                'optimizer': 'adadelta',
                'epochs': 200,
                'learning_rate' : 0.2,
                'batch_size': 500,
                'dropout' : 0.5
                }
        else:
            assert 0, "Bad Config creation: " + self.config.name
            
    def reformat_target(self):
        '''reformat target variable as float32'''
        self.x_train, self.y_train = self.data_factory.get_train_data()
        self.x_test, self.y_test = self.data_factory.get_test_data()
        self.y_train_list = []
        self.y_test_list = []
        for entry in self.y_test:
            if entry == 1:
                self.y_test_list.append([1, 0, 0])
            elif entry == -1:
                self.y_test_list.append([0, 1, 0])
            elif entry == 0:
                self.y_test_list.append([0, 0, 1])

        for entry in self.y_train:
            if entry == 1:
                self.y_train_list.append([1, 0, 0])
            elif entry == -1:
                self.y_train_list.append([0, 1, 0])
            elif entry == 0:
                self.y_train_list.append([0, 0, 1])

        self.y_train_list = np.array(self.y_train_list).astype('float32')
        self.y_test_list = np.array(self.y_test_list).astype('float32')

    def train(self):
        '''train on the dataset provided'''

        self.set_params()
        self.reformat_target()

        ## The Neural Network:
        self.model = Sequential()
        # Input - Layer
        self.model.add(layers.Dense(64, activation='relu', input_shape=(self.x_train.shape[1], )))
        # Hidden - Layers
        self.model.add(layers.Dropout(self.model_params['dropout'], noise_shape=None, seed=None))
        self.model.add(layers.Dense(48, activation='relu'))
        self.model.add(layers.Dropout(self.model_params['dropout'], noise_shape=None, seed=None))
        self.model.add(layers.Dense(24, activation='relu'))
        self.model.add(layers.Dropout(self.model_params['dropout'], noise_shape=None, seed=None))
        self.model.add(layers.Dense(12, activation='relu'))
        self.model.add(layers.Dropout(self.model_params['dropout'], noise_shape=None, seed=None))
        self.model.add(layers.Dense(6, activation='relu'))
        # Output- Layer
        self.model.add(layers.Dense(3, activation='softmax'))
        self.model.summary()

        # Compile the network with specified optimizer, loss and  metrics.
        self.model.compile(optimizer=self.model_params['optimizer'],
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # Fitting the model
        history = self.model.fit(self.x_train, self.y_train_list,
                                 epochs=self.model_params['epochs'],
                                 batch_size=self.model_params['batch_size'],
                                 validation_data=(self.x_test, self.y_test_list))
        print(history)

    def test(self):
        '''test and return confusion_matrix and classification_report'''

        test_preds = self.model.predict(self.x_test)

        #Classification Report

        preds_list = []
        target_list = []
        for i in range(len(test_preds)):
            preds_list.append(np.argmax(test_preds[i]))
        for i in range(len(self.y_test_list)):
            target_list.append(np.argmax(self.y_test_list[i]))

        cr_nn = classification_report(target_list, preds_list)

        ## Confusion Matrix

        cm_nn = confusion_matrix(target_list, preds_list)
        plt.imshow(cm_nn, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(3)
        plt.xticks(tick_marks, ['buy', 'sell', 'hold'], rotation=45)
        plt.yticks(tick_marks, ['buy', 'sell', 'hold'])
        plt.ylabel('Target')
        plt.xlabel('Predictions')
        plt.tight_layout()

        return cm_nn, cr_nn
