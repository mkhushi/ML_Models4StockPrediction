
## ASX:APT only

"""
For state-space training and testing
"""

import glob
import random

from DataFactory.dataset import DataSet

class AsxDataSingle(DataSet):
    '''Dataset to train and test on single random stock of ASX200'''

    def __init__(self, is_classifier=False):

        shift = 1
        price_change = 0.5
        self.seed = 6
        random.seed(self.seed)
        super(AsxDataSingle, self).__init__(is_classifier, 'adjusted_close', shift, price_change)

        data_path = self.get_realpath('ASX_ASX_*')

        files = glob.glob(data_path)

#        train_stock_count = 1
#        test_stock_count = 1
#        CPU, XRO

        subset_files = [files[52], files[199]]

        subset_test_files = [subset_files[i] for i in [1]]

        for file in subset_test_files:
            subset_files.remove(file)

        self.date_col = 'date'
        self.train_data = self.fetch_dataset(subset_files)
        self.test_data = self.fetch_dataset(subset_test_files)

        self.process_data()
