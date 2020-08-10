# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 13:51:52 2018

@author: aditya.sharma
"""

#Logistic Regression from sklearn
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from ModelFactory.model import Model, Configs

class LR(Model):
    '''logistic regression model'''

    def set_params(self):
        '''prepares the hypermarameters to be used by the model'''

        if self.config is Configs.Config1:
            self.model_params = {
                'solver':'lbfgs',
                'multi_class':'multinomial',
                'C':1,
                'penalty':'l2',
                'fit_intercept':True,
                'max_iter':100,
                'random_state':1
                }
        elif self.config is Configs.Config2:
            self.model_params = {
                'solver':'newton-cg',
                'multi_class':'multinomial',
                'C':1,
                'penalty':'l2',
                'fit_intercept':True,
                'max_iter':100,
                'random_state':1
                }
        elif self.config is Configs.Config3:
            self.model_params = {
                'solver':'liblinear',
                'multi_class':'multinomial',
                'C':1,
                'penalty':'l1',
                'fit_intercept':True,
                'max_iter':100,
                'random_state':1
                }
        elif self.config is Configs.Config4:
            self.model_params = {
                'solver':'saga',
                'multi_class':'multinomial',
                'C':1,
                'penalty':'l1',
                'fit_intercept':True,
                'max_iter':100,
                'random_state':1
                }
        else:
            assert 0, "Bad Config creation: " + self.config.name
            
    def train(self):
        '''train on the dataset provided'''

        self.set_params()
        x_train, y_train = self.data_factory.get_train_data()

        self.model = LogisticRegression(**self.model_params)
        self.model.fit(x_train, y_train)



    def test(self):
        '''test and return confusion_matrix and classification_report'''

        x_test, y_test = self.data_factory.get_test_data()
        y_pred = self.model.predict(x_test)
        print("LR: Predictions have finished")

        correct_lr = [1 if a == b else 0 for (a, b) in zip(y_pred, y_test)]
        accuracy_lr = (sum(map(int, correct_lr)) / float(len(correct_lr)))
        print('accuracy_LR = {0}%'.format(accuracy_lr * 100))

        cm_lr = confusion_matrix(y_test, y_pred)
        cr_lr = classification_report(y_test, y_pred)

        return cm_lr, cr_lr
