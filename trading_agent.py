# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 13:05:41 2018

@author: aditya.sharma
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from ModelFactory.model_factory import model_factory
from ModelFactory.model import ModelName, Configs
from DataFactory.dataset import Experiment

class TradingAgent:  # pylint: disable=too-few-public-methods
    '''class to use models to train and test on specified dataset'''

    def __init__(self, modelName, config, experiment):
        self.model = model_factory(modelName, config, experiment)


    def perform_experiment(self):
        '''salls model cretated during init.
        Trains and then provides test results'''
        print("Start train")
        self.model.train()
        print("Start test")
        confusion_matrix, classification_report = self.model.test()
        print(classification_report)
        
        #plot cm
        o_acc=np.around(np.sum(np.diag(confusion_matrix))/\
                        np.sum(confusion_matrix)*100,1)
        plt.title(f'Confusion Matrix \n Accuracy={o_acc}%', size=18)
        sn.heatmap(confusion_matrix, fmt=".0f", annot=True, cbar=False, 
                   annot_kws={"size":15}, xticklabels=['Sell','Hold','Buy'],
                   yticklabels=['Sell','Hold','Buy'], cmap=plt.cm.Blues)
        plt.xlabel('Predicted Label', size=15);plt.ylabel('True Label', size=15)
        plt.imshow(confusion_matrix, cmap=plt.cm.Blues);plt.colorbar()
        
        
        #For SS
#        MSE, RMSE  = self.model.test()
#        print(MSE, RMSE)


def main():
    ''' perform tests by calling trading agents with different set of:
        models, configurations and experiments'''
        
#    trading_agent = TradingAgent(ModelName.SS,
#                                 Configs.Config1,
#                                 Experiment.Index_Rgr)
    trading_agent = TradingAgent(ModelName.LSTM,
                                 Configs.Config2,
                                 Experiment.Index_Clf)

    trading_agent.perform_experiment()

if __name__ == "__main__":
    main()
