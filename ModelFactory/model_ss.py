## Pote Pongchaikul
## State-space model

"""
References:


* https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html#numpy.ndarray.astype, accessed 21 September
* https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.flatten.html
* https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.tolist.html (accessed 21 September)
* https://pydlm.github.io/installation.html, accessed 5 September
* https://docs.scipy.org/doc/numpy/reference/generated/numpy.nan_to_num.html
* [Hyndman], p38 - 39

"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pydlm import dlm, dynamic

from ModelFactory.model import Model

class SS(Model):
    '''state space model'''

    def __init__(self, config, experiment):
        super(SS, self).__init__(config, experiment)
        self.our_state = None

    def prepare_train(self):
        '''prepare data to be used during training of state-space model'''

        x_train, y_train = self.data_factory.get_train_data()
        response_train = [[r] for r in y_train]

        # Idea of using .tolist():
        ## https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.tolist.html

        # Convert data type to float is via astype()
        ## https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html#numpy.ndarray.astype
        feature_matrix_dd_train = np.matrix(x_train).astype("float").tolist()
        return feature_matrix_dd_train, response_train

    def prepare_test(self):
        '''prepare data to be used during testing of state-space model'''

        x_test, y_test = self.data_factory.get_test_data()
        response_test = [[r] for r in y_test]

        # Idea of using .tolist() https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.tolist.html
        feature_matrix_dd_test = np.matrix(x_test).astype("float").tolist()
        return feature_matrix_dd_test, response_test

    def train(self):
        '''train on the dataset provided'''
        feat_train, resp_train = self.prepare_train()

        # Main package used for State-space model with Kalman filtering (DLM)
        ## https://pydlm.github.io/installation.html.
        
        ## mydlm is created by constructing a dlm object, and model the states using the 
        ## "dynamic" function
        mydlm = dlm(resp_train) + dynamic(features=feat_train, discount=1, name="train")
        
        ## Execute Kalman filtering
        mydlm.fitForwardFilter()
        
        ## Get the weights (the coefficients)
        latent_coef_mean = mydlm.getLatentState(filterType="forwardFilter")
        latent_coef_var = mydlm.getLatentCov(filterType="forwardFilter")
        print(latent_coef_var)
        
        ## It was proved in the report that the latest state, together with test data, are enough
        ## for predictions
        self.our_state = latent_coef_mean[-1]

    def test(self):
        '''test and return prediction MSE and RMSE'''
        feat_test, resp_test = self.prepare_test()
        if np.shape(self.our_state)[0] == np.shape(feat_test)[1]:

            # Flattening array via .flatten()
            ## https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.flatten.html
            pred_test_train = list(np.dot(feat_test, self.our_state).flatten())
            
            ## In case missing values were output from pydlm above, this is the backup
            if np.sum(np.isnan(pred_test_train)) > 0:

                # Converting nan to numbers via numpy nan_to_num
                # https://docs.scipy.org/doc/numpy/reference/generated/numpy.nan_to_num.html

                ## Getting last element of a list:
                ## https://stackoverflow.com/questions/930397/getting-the-last-element-of-a-list-in-python
                _, resp_train = self.prepare_train()

                pred_test_train = resp_train[-1]*len(resp_test)
        else:
            pred_test_train = resp_train[-1]*len(resp_test)

        ## "Natural choice" as mentioned in the report
        error_sq = [(p-t)**2 for p, t in zip(pred_test_train, resp_test)]
        mse = np.mean(error_sq)
        rmse = np.sqrt(mse)
        return mse, rmse

    def naive(self):
        '''Naive method - see [Hyndman] Forecasting, Principles and Practice, p38 - 39'''
        
        ## Take the last observation, and use that for the rest of the test data
        _, resp_train = self.prepare_train()
        _, resp_test = self.prepare_test()
        last_obs = list(np.array(resp_train).flatten())[-1]
        error_sq_naive = (last_obs-resp_test)**2
        mse_naive = np.mean(error_sq_naive)
        rmse_naive = np.sqrt(mse_naive)
        return mse_naive, rmse_naive

    def average_method(self):
        '''Average method - see [Hyndman] Forecasting, Principles and Practice, p38 - 39'''
        
        ## Average the entire training data, and use this as forecast
        _, resp_train = self.prepare_train()
        _, resp_test = self.prepare_test()
        avg_obs = np.mean(resp_train)
        error_sq_avg = (avg_obs-resp_test)**2
        mse_avg = np.mean(error_sq_avg)
        rmse_avg = np.sqrt(mse_avg)
        return mse_avg, rmse_avg

    def get_state_names(self):
        '''get names of states used during training'''
        name_states = {}
        states = ['Open', 'High', 'Low', 'Close', 'Volume', 'GDP-Australia',\
     'GDP-China',\
     'GDP per capita-Australia',\
     'GDP per capita-China',\
     'Industry-Australia',\
     'Industry-China',\
     'Inflation-Australia',\
     'Inflation-China',\
     'Manufacturing-Australia',\
     'Manufacturing-China',\
     'Total reserves-Australia',\
     'Total reserves-China',\
     'Trade-Australia', 'Trade-China', 'SMA', 'WMA',\
     'Momentum', 'RSI', 'Sto_K', 'Sto_D', 'Wil_R', 'MACD', 'CCI', 'T_SMA', 'T_WMA'\
     'T_Sto_K', 'T_Sto_D', 'T_Wil_R', 'T_MACD', 'T_RSI', 'T_CCI', 'T_AD',\
     'T_Momentum', 'OBV', 'SMA_5', 'Bias', 'PSY', 'Move1', 'T_Move1', 'High1', 'Low1',\
     'Move2', 'T_Move2', 'High2', 'Low2', 'Move3', 'T_Move3', 'High3', 'Low3', 'Move4',\
     'T_Move4', 'High4', 'Low4', 'Move5', 'T_Move5', 'High5', 'Low5', 'EWMA-0.25',\
     'EWMA-0.5', 'EWMA-0.75', 'Volatility-100', 'Volatility-150', 'Volatility-200',\
     'Volatility-250', 'autocorrelation', 'Volume_MA-100', 'Volume_MA-150',\
     'Volume_MA-200', 'Volume_MA-250']
        for i,_ in enumerate(states):
            name_states[states[i]] = self.our_state[i]
            print(i, states[i])

        return name_states

    def plot_series(self):
        '''visualisation of states (weights, regarded as importance for each predictors)'''
        _, resp_train = self.prepare_train()
        _, resp_test = self.prepare_test()
        st_names = self.get_state_names()
        plt.figure(figsize=(8, 4))
        plt.plot(resp_train)
        plt.title("Training data")
        plt.xlabel("Time (days)")
        print("\n")
        plt.figure(figsize=(8, 4))
        plt.plot(resp_test)
        plt.title("Test data")
        plt.xlabel("Time (days)")
        plt.show()
        plt.figure(figsize=(8, 15))
        sns.barplot(y=list(st_names.keys()), x=list(st_names.values()), orient="h")
        plt.title("States")
        plt.show()
