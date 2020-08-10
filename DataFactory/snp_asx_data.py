# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 14:15:05 2018

@author: aditya.sharma
"""
import glob
import random

from DataFactory.dataset import DataSet

class SnpAsxData(DataSet):
    '''Dataset to train on stocks of S&P500 and test on random stocks of ASX200'''

    def __init__(self, is_classifier=True):

        shift = 1
        price_change = 0.5
        self.seed = 6
        random.seed(self.seed)

        super(SnpAsxData, self).__init__(is_classifier, 'adjusted_close', shift, price_change)

        train_stock_count = 30
        test_stock_count = 5

        train_data_path_1 = self.get_realpath('SP500_1_*')
        train_data_path_2 = self.get_realpath('SP500_2_*')
        train_data_path_3 = self.get_realpath('SP500_3_*')
        test_data_path = self.get_realpath('ASX_ASX_*')

        self.date_col = 'date'
        train_files = glob.glob(train_data_path_1)
        train_files += glob.glob(train_data_path_2)
        train_files += glob.glob(train_data_path_3)
        test_files = glob.glob(test_data_path)

        subset_train_files = [train_files[i] for i in random.sample(range(len(train_files)),
                                                                    train_stock_count)]

        subset_test_files = [test_files[i] for i in random.sample(range(len(test_files)),
                                                                  test_stock_count)]

        self.train_data = self.fetch_dataset(subset_train_files)
        self.test_data = self.fetch_dataset(subset_test_files)

        self.process_data()
