  # -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 13:51:52 2018

@author: aditya.sharma
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import CuDNNLSTM
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.sequence import TimeseriesGenerator

from ModelFactory.model import Model, Configs


class LSTM(Model): # pylint: disable=too-many-instance-attributes
    '''LSTM model'''

    def __init__(self, config, experiment):
        super(LSTM, self).__init__(config, experiment)
        self.opt = None
        self.checkpoint = None
        self.tensorboard = None
        self.early_stopping = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.weights = None
        self.train_data_gen = None
        self.test_data_gen = None

    def set_params(self):
        '''prepares the hypermarameters to be used by the model'''

        if self.config is Configs.Config1:
            self.model_params = {
                'optimizer':Adam,
                'epochs':150,
                'learning_rate':0.001,
                'decay':1e-6,
                'dropout':0.5,
                'SEQ_LEN':90,
                'NAME':f"LSTM-{self.config}-{time.strftime('%Y-%m-%d %H-%M-%S')}",
                'patience':100,
                'lstm_neurons':[256, 256, 128],
                'shuffle':True,
                'batch_size':128,
                'steps_per_epoch':50
                }
        elif self.config is Configs.Config2:
            self.model_params = {
                'optimizer':Adam,
                'epochs':150,
                'learning_rate':0.001,
                'decay':1e-6,
                'dropout':0.5,
                'SEQ_LEN':10,
                'NAME':f"LSTM-{self.config}-{time.strftime('%Y-%m-%d %H-%M-%S')}",
                'patience':100,
                'lstm_neurons':[256, 256, 128],
                'shuffle':True,
                'batch_size':128,
                'steps_per_epoch':50
                }
        else:
            assert 0, "Bad Config creation: " + self.config.name
    @staticmethod
    def get_weights(dependent_var):
        '''calculate classification weights'''
        classes, cnt = np.unique(dependent_var, return_counts=True, axis=0)
        weights = 1/(cnt/cnt.sum())
        weights = weights/weights.sum()
        
        return dict(zip(classes, weights))

    def add_optimizer(self):
        '''add optimiser to be used by LSTM'''
        self.opt = self.model_params['optimizer'](lr=self.model_params['learning_rate'],
                                                  decay=self.model_params['decay'],
                                                  clipnorm=1.)


    def create_model(self):
        '''create LSTM model'''

        self.model = Sequential()

        self.model.add(CuDNNLSTM(self.model_params['lstm_neurons'][0],
                                 return_sequences=True,
                                 input_shape=(self.model_params['SEQ_LEN'],
                                              self.x_train.shape[1])))
        self.model.add(Dropout(self.model_params['dropout']))
        self.model.add(BatchNormalization())

        if len(self.model_params['lstm_neurons']) > 1:
            for i in self.model_params['lstm_neurons'][1:]:

                self.model.add(CuDNNLSTM(i, return_sequences=[True if i != \
                              self.model_params['lstm_neurons'][-1] else False]))
                self.model.add(Dropout(self.model_params['dropout']))
                self.model.add(BatchNormalization())

        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(self.model_params['dropout']))

        self.model.add(Flatten())
        self.model.add(Dense(3, activation='softmax'))

        self.add_optimizer()

        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=self.opt, metrics=['accuracy'])
        self.model.summary()
        self.tensorboard = TensorBoard(log_dir=f"LSTM_Models/LSTM_logs/\
                                       {self.model_params['NAME']}")
        #run tensorboard from the console with next comment to follow training 
        #tensorboard --logdir=LSTM_Models/LSTM_logs/

        self.checkpoint = ModelCheckpoint('LSTM_Models/Models/LSTM_T1-Best',
                                          monitor='val_acc', verbose=1,
                                          save_best_only=True, mode='max')

        self.early_stopping = EarlyStopping(monitor='val_loss',
                                            patience=self.model_params['patience'])

    def input_data(self):
        '''timeseries data creation'''

        self.x_train, self.y_train = self.data_factory.get_train_data()
        self.x_test, self.y_test = self.data_factory.get_test_data()

        self.y_train.replace({1:2, 0:1, -1:0}, inplace=True)
        self.y_test.replace({1:2, 0:1, -1:0}, inplace=True)

        self.weights = self.get_weights(self.y_train)
        
        self.train_data_gen = TimeseriesGenerator(np.array(self.x_train, 
                                                           dtype=np.float32),
                                                  self.y_train.values,
                                                  length=self.model_params['SEQ_LEN'])
        self.test_data_gen = TimeseriesGenerator(np.array(self.x_test, 
                                                          dtype=np.float32),
                                                 self.y_test.values,
                                                 length=self.model_params['SEQ_LEN'])


    def train(self):
        '''train on the dataset provided'''

        self.set_params()

        self.input_data()

        self.create_model()
        
        self.ttl_batches=int((len(self.x_train)-self.model_params['SEQ_LEN'])/\
                             self.model_params['batch_size'])
        
        history = \
        self.model.fit_generator(generator=self.train_data_gen,
                                 epochs=self.model_params['epochs'],
                                 shuffle=self.model_params['shuffle'],
                                 steps_per_epoch=\
                                 np.min([self.ttl_batches,
                                         self.model_params['steps_per_epoch']]),                                                        
                                 validation_data=self.test_data_gen,
                                 class_weight=self.weights,
                                 callbacks=[self.tensorboard,
                                            self.checkpoint,
                                            self.early_stopping])
        print(history)

    def test(self):
        '''test and return confusion_matrix and classification_report'''

        self.model.load_weights('LSTM_Models/Models/LSTM_T1-Best')

        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=self.opt, metrics=['accuracy'])

        y_lstm_pred = self.model.predict_generator(self.test_data_gen)
        y_lstm_pred = np.argmax(y_lstm_pred, axis=1)

        print("LSTM: Predictions have finished")

        cm_lstm = confusion_matrix(self.y_test[self.model_params['SEQ_LEN']:],
                                   y_lstm_pred)
        o_acc=np.around(np.sum(np.diag(cm_lstm))/np.sum(cm_lstm)*100,1)
        
        plt.title(f'Confusion Matrix \n Accuracy={o_acc}%', size=18)
        sn.heatmap(cm_lstm, fmt=".0f", annot=True, cbar=False, 
                   annot_kws={"size":15}, xticklabels=['Sell','Hold','Buy'],
                   yticklabels=['Sell','Hold','Buy'])
        plt.xlabel('Predicted Label', size=15);plt.ylabel('True Label', size=15)
        print(np.diag(cm_lstm).sum())

        cr_lstm = classification_report(self.y_test[self.model_params['SEQ_LEN']:],
                                        y_lstm_pred)

        return cm_lstm, cr_lstm
