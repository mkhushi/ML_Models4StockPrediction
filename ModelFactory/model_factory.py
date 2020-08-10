# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 14:28:38 2018

@author: aditya.sharma
"""
from ModelFactory.model import ModelName
from ModelFactory.model_gb import GB
from ModelFactory.model_ss import SS
from ModelFactory.model_lstm import LSTM
from ModelFactory.model_lr import LR
from ModelFactory.model_nn import NN

def model_factory(model_name, config, experiment):
    '''returns model depending upon the model-name
    models are created using the config and experement argument'''

    if model_name is ModelName.GB:
        model = GB(config, experiment)
    elif model_name is ModelName.SS:
        model = SS(config, experiment)
    elif model_name is ModelName.LSTM:
        model = LSTM(config, experiment)
    elif model_name is ModelName.LR:
        model = LR(config, experiment)
    elif model_name is ModelName.NN:
        model = NN(config, experiment)
    else:
        assert 0, "Bad Model creation: " + model_name.name

    return model
