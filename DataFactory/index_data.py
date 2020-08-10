# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 14:15:05 2018

@author: aditya.sharma
"""
import pandas as pd

from DataFactory.dataset import DataSet

class IndexData(DataSet):
    '''Dataset to train on S&P500 Index and test on ASX200 Index'''

    def __init__(self, is_classifier=True):

        shift = 10
        price_change = 2

        super(IndexData, self).__init__(is_classifier, 'Adj Close', shift, price_change)

        train_data_path = self.get_realpath('SP500_index_SP500_index.csv')
        test_data_path = self.get_realpath('ASX200_index_ASX200_index.csv')


        self.train_data = self.pre_process_data(pd.read_csv(train_data_path,
                                                            index_col=False))
        self.test_data = self.pre_process_data(pd.read_csv(test_data_path,
                                                           index_col=False))

        self.process_data()
