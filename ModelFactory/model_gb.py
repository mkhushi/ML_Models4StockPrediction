# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 13:51:52 2018

@author: aditya.sharma
"""
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
#from imblearn.pipeline import Pipeline
#from imblearn.over_sampling import SMOTE

from ModelFactory.model import Model, Configs

class GB(Model):
    '''Gradient boosting model'''

    def set_params(self):
        '''prepares the hypermarameters to be used by the model'''
        if self.config is Configs.Config1:
            self.model_params = {
                'n_estimators': 500,
                'max_features': 0.1,
                'learning_rate' : 0.2,
                'max_depth': 15,
                'min_samples_leaf': 2,
                'subsample': 1,
                #'max_features' : 'sqrt',
                'random_state' : 0,
                'verbose': 0
                }
        elif self.config is Configs.Config2:
            self.model_params = {
                'n_estimators': 100,
                'max_features': 0.1,
                'learning_rate' : 0.2,
                'max_depth': 15,
                'min_samples_leaf': 2,
                'subsample': 1,
                #'max_features' : 'sqrt',
                'random_state' : 0,
                'verbose': 0
                }
        elif self.config is Configs.Config3:
            self.model_params = {
                'n_estimators': 50,
                'max_features': 0.1,
                'learning_rate' : 0.2,
                'max_depth': 15,
                'min_samples_leaf': 2,
                'subsample': 1,
                #'max_features' : 'sqrt',
                'random_state' : 0,
                'verbose': 0
                }
        else:
            assert 0, "Bad Config creation: " + self.config.name

    def train(self):
        '''train on the dataset provided'''
        self.set_params()
        print("GB: training with", self.config.name)

        x_train, y_train = self.data_factory.get_train_data()
        #print("GB: training with row-count", len(x_train.index))

        self.model = GradientBoostingClassifier(**self.model_params)
        # Fit the model to our SMOTEd train and target
        #gb.fit(smote_train, smote_target)
        self.model.fit(x_train, y_train)

    def test(self):
        '''test and return confusion_matrix and classification_report'''
        x_test, y_test = self.data_factory.get_test_data()
        y_gb_pred = self.model.predict(x_test)
        print("GB: Predictions have finished")

        correct_gb = [1 if a == b else 0 for (a, b) in zip(y_gb_pred, y_test)]
        accuracy_gb = (sum(map(int, correct_gb)) / float(len(correct_gb)))
        print('accuracy_gb = {0}%'.format(accuracy_gb * 100))

        cm_gb = confusion_matrix(y_test, y_gb_pred)
        cr_gb = classification_report(y_test, y_gb_pred)

        return cm_gb, cr_gb
