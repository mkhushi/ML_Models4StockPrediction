

## Import packages
import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from pydlm import dlm, dynamic

## Ref: https://pydlm.github.io/pydlm_user_guide.html#modeling, accessed 5 September
from pydlm import dlm, dynamic

from ModelFactory.model import Model, Configs

train = pd.read_csv("https://raw.github.sydney.edu.au/asha7190/MI_FinancialTrading/master/Data/Stocks%20unnormalized/SP500_1_AAPL.csv?token=AAAH4aMML4Go38wop_Tr_ugd7BnltaQKks5bx6g_wA%3D%3D",
                    index_col = 0,parse_dates = True)
test = pd.read_csv("https://raw.github.sydney.edu.au/asha7190/MI_FinancialTrading/master/Data/Stocks%20unnormalized/ASX_ASX_ANZ.csv?token=AAAH4TvThsByHuNSIYn8Ds7J6oWJhK-mks5bx6dvwA%3D%3D",
                      index_col=0, parse_dates=True)


class SS(Model):

    
    def normalize(data):
      ## Standard deviation in Numpy
      ## https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html, accessed 17 September
        return (data - np.mean(data))/np.std(data)

    ## Perform normalization
    
    ## Drop columns:
    ## https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html
    ## accessed 21 September

    def perform_normalize():
        

        X_train = train.drop(["adjusted_close","symbol"],axis = "columns")
        y_train = train.adjusted_close
        
        X_test = test.drop(["adjusted_close","symbol"],axis = "columns")
        y_test = test.adjusted_close
        
        for cc in X_train.columns:
            X_train.loc[:,cc] = SS.normalize(X_train.loc[:,cc])
        for dd in X_test.columns:
            X_test.loc[:,dd] = SS.normalize(X_test.loc[:,dd])
        
        y_train = SS.normalize(y_train) 
        y_test = SS.normalize(y_test)
    
        return X_train, y_train, X_test, y_test

    
    def prepare_train(): 
        X_train, y_train, X_test, y_test = SS.perform_normalize()
        response_train = [[r] for r in y_train]
        feature_matrix_dd_train  = np.matrix(X_train).astype("float").tolist()
        return feature_matrix_dd_train, response_train
        
    def prepare_test(): 
        #X_test , y_test = self.data_factory.get_test_data()
        response_test = [[r] for r in y_test]
        feature_matrix_dd_test  = np.matrix(X_test).astype("float").tolist()
        return feature_matrix_dd_test, response_test
        
    def train(): 
        feat_train, resp_train = SS.prepare_train()
        mydlm = dlm(resp_train) + dynamic(features = feat_train,discount = 1,name = "train")
        mydlm.fitForwardFilter()
        latent_coef_mean = mydlm.getLatentState(filterType = "forwardFilter")
        latent_coef_var = mydlm.getLatentCov(filterType = "forwardFilter")
        our_state = latent_coef_mean[len(latent_coef_mean)-1]
        return our_state
    
    def test(): 
        feat_train, resp_train = SS.prepare_train()
        our_state = SS.train()
        feat_test, resp_test = SS.prepare_test()
        pred_test_train = []
        if np.shape(our_state)[0] == np.shape(feat_test)[1]:
            pred_test_train = list(np.dot(feat_test,our_state).flatten())
            if np.sum(np.isnan(pred_test_train) == True) > 0:
              pred_test_train = resp_train[len(resp_train) - 1] + np.nan_to_num(pred_test_train)
        else:
            pred_test_train = np.nan_to_num(our_state) + resp_train[len(resp_train) - 1]
            
       
        error_sq = [(p-t)**2 for p,t in zip(pred_test_train,resp_test)]
        MSE = np.mean(error_sq)
        RMSE = np.sqrt(MSE)
        return pred_test_train, error_sq, MSE, RMSE
    
    
pred_test_train, error_sq, MSE, RMSE = SS.test()
print(np.mean(error_sq[:5]))
