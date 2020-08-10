# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 10:46:22 2018

@author: aditya.sharma
"""
from enum import Enum, auto
from DataFactory.data_factory import data_factory

class ModelName(Enum):
    '''Enum for model names'''
    GB = auto()
    LSTM = auto()
    NN = auto()
    LR = auto()
    SS = auto()

class Configs(Enum):
    '''Enum for configurations to be used for models.
    Config is set of hyperparameters to be used by different models.
    Each model defines it own config parameters'''

    Config1 = auto()
    Config2 = auto()
    Config3 = auto()
    Config4 = auto()
    Config5 = auto()
    Config6 = auto()
    Config7 = auto()
    Config8 = auto()

class Model:  # pylint: disable=too-few-public-methods
    '''base class for all models'''

    def __init__(self, config, experiment):
        self.model_params = None
        self.model = None
        self.config = config
        self.data_factory = data_factory(experiment)
