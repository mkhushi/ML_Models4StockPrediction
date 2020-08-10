Readme file to explain Machine Learning code

Project members:



Software: Coded in Spyder 3.6, Python 3, 

Files:

Start in code folder:

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

5. 



