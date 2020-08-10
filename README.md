# ML_Models4StockPrediction
Code Implementation of a range of supervised machine learning methods include Gradient Boosting, Logistic Regression, State Space Modeling, Deep Neural Networks and LSTMs.

1. trading_agent.py - start file that pulls all the objects to run models.  This is the initial file.  Important to set the working directory properly.

This is the main program to start all the codes running.  Parameters to be defined under trading agent function (example below):

    trading_agent = TradingAgent(ModelName.LSTM,
                                 Configs.Config1,
                                 Experiment.Index_Clf)


a. ModelName.LSTM: python file under directory "ModelFactory"
b. Configs.Config1: which kind of Configuration of the model is to be used.  Config1 is defined under ModelName.LSTM.  All configurations are contained in the ModelName.LSTM file.
c. Experiment.Index_Clf: Type of experimento to run.

Some of the configurations 

Data collection folder:

2. get_price_data.py - get pricing data from alpha vantage.  There's a limit in the number of daily queries that alpha vantage accepts of 5 queries per minute and 500 queries per day.  Input parameters are the stocks to be downloaded and which parameters of each stock

3. get_wb_data.py - get world bank macroeconomic data from the world bank


Feature engineering folder:

4. index_data.py - creates .CSV files for each stock.  It derives various features from pricing data.  It also derives binomial information from candlestick charts, such as the many technical patterns, e.g. hammer, shooting star, morning star, bullish abandoned baby.


Data factory:

These files are used to configure dependent data for the experiments


##Abstract:##
People have always tried to forecast future asset prices to make gains. The rise of modern stock markets, with large
data gathering capabilities, has opened a range of theories to help an investor forecast future prices. These range from
fundamental analysis in which one values a security according to its underlying value to technical analysis through which the
future price is predicted from its past time series data. The effect of many different price forming expectations is that stock
markets are a very dynamic and crowded system. Time-series of past stock price data encodes a lot of information which
has to be extracted through extensive use of feature engineering. Three main types of feature engineering are proposed here:
(i) economic (fundamental) data, (ii) time series transformations and (iii) technical analysis feature extraction. Machine
learning models can be successfully trained to learn from past data in order to predict future prices. Using machine learning
output combined with expert knowledge and having clear risk minimisation strategies such as stop-loss and lock-gain
strategies one can increase his portfolio return. Machine learning is a good tool to augment a trader’s prediction accuracy.
The system designed for this report can be easily extended to accept further economic and fundamental data and to accept
further feature engineering. Different experiments can be easily trained and reports can be easily calculated for different
machine learning techniques. This paper proposes an extensive range of supervised machine learning methods to process
the engineered features into a solution for the stock price forecasting problem. The range of supervised learning methods
include Gradient Boosting, Logistic Regression, State Space Modeling, Deep Neural Networks and LSTMs. The suggested
implementation and specific architecture for these methods given the financial context is also discussed. Additionally insight
will be given into the resources required to tackle the stock price forecasting problem and the allocated schedule for the
project complete with expected outcomes. Solving this problem has motivation not just in the financial sense but also in the
academic sphere where it has been debated since the emergence of the markets themselves.

