# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 14:33:13 2018

@author: aditya.sharma
"""

from DataFactory.dataset import Experiment
from DataFactory.index_data import IndexData
from DataFactory.snp_asx_data import SnpAsxData
from DataFactory.snp_data import SnpData
from DataFactory.asx_data import AsxData

from DataFactory.asx_data_single import AsxDataSingle

def data_factory(experiment):
    '''create dataset as per the experiment'''
    dataset = None

    if experiment == Experiment.Index_Clf:
        dataset = IndexData(is_classifier=True)
    elif experiment == Experiment.Index_Rgr:
        dataset = IndexData(is_classifier=False)
    elif experiment == Experiment.SnP_ASX_Clf:
        dataset = SnpAsxData(is_classifier=True)
    elif experiment == Experiment.SnP_ASX_Rgr:
        dataset = SnpAsxData(is_classifier=False)
    elif experiment == Experiment.SnP_Clf:
        dataset = SnpData(is_classifier=True)
    elif experiment == Experiment.SnP_Rgr:
        dataset = SnpData(is_classifier=False)
    elif experiment == Experiment.ASX_Clf:
        dataset = AsxData(is_classifier=True)
    elif experiment == Experiment.ASX_Rgr:
        dataset = AsxData(is_classifier=False)
    elif experiment == Experiment.ASX_data_single:
        dataset = AsxDataSingle(is_classifier=False)
    else:
        assert 0, "Bad Data creation: " + experiment.name

    return dataset
