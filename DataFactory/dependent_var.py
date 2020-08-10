# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:28:02 2018

@author: aditya.sharma
"""
#Load libraries
import pandas as pd
import numpy as np

def get_price_change(price_data, #pylint: disable-msg=R0914
                     days,
                     percentage_change,
                     price_col='adjusted_close',
                     is_index=False):
    """returns 1, 0 -1 for percentage change in price over the different set
    of days.
    Input:
    price_data: dataset which should have columns: symbol and adjusted_close
    days: Tuple(in ascending order) of number of days we need to consider
    while calculating the price change
    percentage_change: Tuple(in ascending order) of percentage change to check
    shift: the number of days in future to predict the change. Shift = 1 means
    the output variable will indicate price movement on T+1 day as per the
    price_col: price change today
    column name to look for price change
    is_index: ignores symbol information if datset is for an index

    Output:
    price_data + new columns to indicate price change,
    e.g. column price_3_5 means price change w.r.t price of T-3 days and check
    if the price has changed 5%. 1: >= 5%, -1: <= 5%, 0 if [-5%, 5%]

    """

    cols = list()
    df_price = pd.DataFrame(price_data)

    for day in days:
        column_name = "price_change_percent_"+str(day)
        df_price[column_name] = 0.0
        cols.append(column_name)
        for price_change in percentage_change:
            column_name = "price_"+str(day)+"_"+str(price_change)
            df_price[column_name] = 0
            cols.append(column_name)

    price_change_df = pd.DataFrame(0, index=np.arange(len(price_data)), columns=cols)

    for index, data in df_price.iterrows():
        for day in days:
            prev_index = index - day
            if index <= 0 or prev_index < 0:
                break
            elif is_index or data['symbol'] == df_price.loc[prev_index]['symbol']:
                prev_price = df_price.loc[prev_index][price_col]
                current_price = data[price_col]
                if prev_price == 0.0:
                    price_change_percent = 0.0
                else:
                    price_change_percent = 100*(current_price - prev_price)/prev_price
# =============================================================================
#                 print("---------------------")
#                 print("day", day)
#                 print("prev_price", prev_price)
#                 print("current_price", current_price)
#                 print("price_change_percent", price_change_percent)
# =============================================================================
                price_change_df.iloc[index]["price_change_percent_"+str(day)] = price_change_percent

                for price_change in percentage_change:
                    if abs(price_change_percent) > price_change:
                        column_name = "price_"+str(day)+"_"+str(price_change)
                        price_change_df.iloc[index][column_name] = np.sign(price_change_percent/price_change)

    price_data.update(price_change_df)

    return price_data
