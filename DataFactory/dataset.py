# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 12:29:00 2018

@author: aditya.sharma
"""

from os.path import dirname, realpath, join
from enum import Enum, auto
import numpy as np
import pandas as pd

from sklearn import preprocessing

from DataFactory.dependent_var import get_price_change

class Experiment(Enum):
    '''Enup for experiments to be performed'''
    Index_Clf = auto() #Experiment to perform classification on Index dataset
    Index_Rgr = auto() #Experiment to perform regression on Index dataset
    SnP_ASX_Clf = auto() #train on S&P and test on ASX
    SnP_ASX_Rgr = auto() #train on S&P and test on ASX
    SnP_Clf = auto() #train on S&P and test on S&P
    SnP_Rgr = auto() #train on S&P and test on S&P
    ASX_Clf = auto() #train on ASX and test on ASX
    ASX_Rgr = auto() #train on ASX and test on ASX
    ASX_data_single = auto()

class DataSet: # pylint: disable=too-many-instance-attributes
    '''base class for datasets to be used by models'''

    def __init__(self, is_classifier, price_col, shift, price_change):

        self.data_dir = join(join(dirname(dirname(realpath(dirname(__file__)))),
                                  'Data'), 'GOLD Stocks unnormalized')

        self.is_classifier = is_classifier
        self.price_col = price_col
        self.shift = shift
        self.price_change = price_change
        self.train_data = None
        self.test_data = None
        self.date_col = 'Date'
        self.remove_symbols = None
        self.col_names = None

    def get_realpath(self, file_name):
        '''get real-path of the file'''
        return join(self.data_dir, file_name)

    def fetch_dataset(self, files):
        '''read files for each stock and create dataset'''
        frames = list()

        if self.is_classifier:
            dependent_col = self.pre_process_data(pd.read_csv(files[0],
                                                              index_col=False)).columns[-1]

        frames = [self.pre_process_data(pd.read_csv(files[i], index_col=False))
                  for i in range(len(files))]

        dataframe = pd.concat(frames, ignore_index=True, sort=False)

        if self.is_classifier:
            col = dataframe.pop(dependent_col)
            dataframe[dependent_col] = col

        return dataframe

    @staticmethod
    def handle_missing_data(data):
        '''handle/impute missing data'''
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(axis='columns', how='all', inplace=True)
        for index, row in data.iterrows():
            if np.around(row.isna().sum()/len(data.columns), 2) > 0.5:
                data.drop(index=index, inplace=True)
        for col in data.columns:
            if np.around(data[col].isna().sum()/len(data[col]), 2) > 0.1:
                data.drop(columns=col, inplace=True)
        data.dropna(axis='index', how='any', inplace=True)
        return data

    def pre_process_data(self, data):
        '''add dependent variable to dataset for classification experiment'''
        if self.is_classifier:
            data = get_price_change(price_data=data,
                                    days=[self.shift],
                                    percentage_change=[self.price_change],
                                    price_col=self.price_col,
                                    is_index=True)

        data.drop(columns=[self.date_col], inplace=True)
        return data


    def process_data(self):
        '''create train and test datasets'''

        train_data = self.handle_missing_data(self.train_data)

        test_data = self.handle_missing_data(self.test_data)

        for col in train_data.columns:
            if col not in test_data.columns:
                train_data.drop(columns=col, inplace=True)
        for col in test_data.columns:
            if col not in train_data.columns:
                test_data.drop(columns=col, inplace=True)

        self.train_data = train_data
        self.test_data = test_data


    def get_data(self, is_train):
        '''get train/test dataset'''

        if is_train:
            data = self.train_data
        else:
            data = self.test_data

        if self.is_classifier:
            dependent_var = data.iloc[:, -1]
            data = data.iloc[:, 0:-1]
        else:
            dependent_var = data[self.price_col]
            data = data.drop(columns=[self.price_col])

        data = data.iloc[:-self.shift, :]
        dependent_var = dependent_var.shift(-self.shift)
        dependent_var = dependent_var.iloc[:-self.shift]
        print(data.symbol.unique(), sep=' ')
        if self.remove_symbols:
            data.drop(columns=['symbol'], inplace=True)
        print('Total records:', len(data))
        self.col_names = data.columns.values

        data = preprocessing.StandardScaler().fit_transform(data)

        return data, dependent_var

    def get_train_data(self, remove_symbols=True):
        '''return train dataset'''
        print('Training on tickers:')
        self.remove_symbols = remove_symbols
        
        return self.get_data(is_train=True)


    def get_test_data(self):
        '''return test dataset'''
        print('Testing on tickers:')
        return self.get_data(is_train=False)
