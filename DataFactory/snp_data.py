# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 14:15:05 2018

@author: aditya.sharma
"""
import glob
import random

from DataFactory.dataset import DataSet

class SnpData(DataSet):
    '''Dataset to train and test on random stocks of S&P500'''

    def __init__(self, is_classifier=True):

        shift = 1
        price_change = 0.5
        self.seed = 6
        random.seed(self.seed)
        super(SnpData, self).__init__(is_classifier, 'adjusted_close', shift, price_change)

        data_path_1 = self.get_realpath('SP500_1_*')
        data_path_2 = self.get_realpath('SP500_2_*')
        data_path_3 = self.get_realpath('SP500_3_*')

        files = glob.glob(data_path_1)
        files += glob.glob(data_path_2)
        files += glob.glob(data_path_3)

        train_stock_count = 30
        test_stock_count = 5
        subset_files = [files[i] for i in random.sample(range(len(files)),
                                                        train_stock_count + test_stock_count)]
        subset_test_files = [subset_files[i] for i in random.sample(range(len(subset_files)),
                                                                    test_stock_count)]
        for file in subset_test_files:
            subset_files.remove(file)

        self.date_col = 'date'
        self.train_data = self.fetch_dataset(subset_files)
        self.test_data = self.fetch_dataset(subset_test_files)

        self.process_data()
