"""This module has the function to get the World Bank Macroeconomic data"""
#World Bank API
# Data Acquisition
# For citation purposes:
#Sherouse, Oliver (2014). Wbdata. Arlington,VA.
#Available from http://github.com/OliverSherouse/wbdata.
# Documentation: https://wbdata.readthedocs.io/en/latest/index.html

#Load libraries
import datetime
import wbdata

#Default input for the function (further down)
COUNTRIES = ['ARG', 'AUS', 'BEL', 'BRA', 'CAN', 'CHE', 'CHL', 'CHN', 'CYM',
             'DEU', 'FRA', 'GBR', 'HKG', 'IDN', 'IND', 'ITA', 'JPN', 'KOR',
             'MEX', 'MYS', 'NLD', 'NZL', 'PER', 'PHL', 'RUS', 'SGP', 'THA',
             'TUR', 'TWN', 'USA', 'VNM', 'ZAF', 'VGB']
INDICATORS = {'NY.GNS.ICTR.CD':'Gross savings (current US$)',
              'NY.GNP.MKTP.PP.CD':'GNI, PPP (current international $)',
              'NY.GDS.TOTL.CD':'Gross domestic savings (current US$)',
              'NY.GDP.PCAP.PP.CD':\
              'GDP per capita, PPP (current international $)',
              'NY.GDP.MKTP.CD':'GDP (current US$)',
              'NY.GDP.DEFL.KD.ZG':'Inflation, GDP deflator (annual %)',
              'DT.DOD.PVLX.CD':'Present value of external debt (current US$)',
              'DT.DOD.DSTC.IR.ZS':'Short-term debt (% of total reserves)',
              'NY.TAX.NIND.CD':'Taxes less subsidies on products (current US$)',
              'GC.DOD.TOTL.GD.ZS':'Central government debt, total (% of GDP)',
              'FP.CPI.TOTL.ZG':'Inflation, consumer prices (annual %)',
              'FI.RES.TOTL.CD':'Total reserves (includes gold, current US$)',
              'NV.IND.TOTL.CD':\
              'Industry (including construction), value added (current US$)',
              'NV.IND.MANF.CD':'Manufacturing, value added (current US$)',
              'NE.TRD.GNFS.ZS':'Trade (% of GDP)',
              'NE.RSB.GNFS.CD':\
              'External balance on goods and services (current US$)',
              'NE.IMP.GNFS.CD':'Imports of goods and services (current US$)',
              'NE.EXP.GNFS.CD':'Exports of goods and services (current US$)',
              'BX.KLT.DINV.CD.WD':\
              'Foreign direct investment, net inflows (BoP, current US$)',
              'BN.GSR.GNFS.CD':\
              'Net trade in goods and services (BoP, current US$)',
              'EP.PMP.SGAS.CD':'Pump price for gasoline (US$ per liter)',
              'EG.USE.ELEC.KH.PC':'Electric power consumption (kWh per capita)',
              'NY.GDP.MINR.RT.ZS':'Mineral rents (% of GDP)',
              'PA.NUS.FCRF':\
              'Official exchange rate (LCU per US$, period average)',
              'FR.INR.RISK':\
              'Risk premium on lending (lending rate minus treasury bill rate, %)',
              'FR.INR.LEND':'Lending interest rate (%)',
              'FM.LBL.BMNY.ZG':'Broad money growth (annual %)',
              'FB.BNK.CAPA.ZS':'Bank capital to assets ratio (%)',
              'CM.MKT.TRNR':\
              'Stocks traded, turnover ratio of domestic shares (%)',
              'CM.MKT.TRAD.GD.ZS':'Stocks traded, total value (% of GDP)',
              'CM.MKT.TRAD.CD':'Stocks traded, total value (current US$)',
              'CM.MKT.LDOM.NO':'Listed domestic companies, total',
              'CM.MKT.LCAP.GD.ZS':\
              'Market capitalization of listed domestic companies (% of GDP)',
              'CM.MKT.LCAP.CD':\
              'Market capitalization of listed domestic companies (current US$)',
              'NV.AGR.TOTL.CD':\
              'Agriculture, forestry, and fishing, value added (current US$)',
              'AG.YLD.CREL.KG':'Cereal yield (kg per hectare)',
              'AG.PRD.CREL.MT':'Cereal production (metric tons)',
              'DT.DOD.DECT.CD.CG':\
              'Total change in external debt stocks (current US$)',
              'DT.NFL.PNGB.CD':'PNG, bonds (NFL, current US$)',
              'DT.AMT.BLAT.CD':'PPG, bilateral (AMT, current US$)',
              'DT.DOD.RSDL.CD':\
              'Residual, debt stock-flow reconciliation (current US$)',
              'IC.EXP.COST.CD':'Cost to export (US$ per container)',
              'IC.BUS.NREG':'New businesses registered (number)',
              'IC.TAX.TOTL.CP.ZS':'Total tax rate (% of commercial profits)',
              'IC.TAX.PRFT.CP.ZS':'Profit tax (% of commercial profits)',
              'TM.VAL.MRCH.CD.WT':'Merchandise imports (current US$)',
              'ST.INT.XPND.CD':\
              'International tourism, expenditures (current US$)',
              'GC.XPN.TRFT.CN':'Subsidies and other transfers (current LCU)',
              'GC.XPN.TOTL.CN':'Expense (current LCU)',
              'GC.TAX.TOTL.GD.ZS':'Tax revenue (% of GDP)',
              'GC.TAX.INTT.RV.ZS':'Taxes on international trade (% of revenue)',
              'GC.TAX.IMPT.ZS':\
              'Customs and other import duties (% of tax revenue)',
              'GC.TAX.EXPT.ZS':'Taxes on exports (% of tax revenue)',
              'GB.XPD.RSDV.GD.ZS':\
              'Research and development expenditure (% of GDP)',
              'SH.MED.PHYS.ZS':'Physicians (per 1,000 people)',
              'SP.DYN.LE00.IN':'Life expectancy at birth, total (years)',
              'SH.UHC.OOPC.25.ZS':'Proportion of population spending more than'\
              '25% of household consumption or income on out-of-pocket health'\
              'care expenditure (%)'}

DATA_DATE = (datetime.datetime(1999, 1, 1), datetime.datetime(2018, 9, 1))

#Get the data
def get_wbdataframe(indicators=INDICATORS, data_date=DATA_DATE,
                    countries=COUNTRIES):
    """
    This function returns a dataframe with the requested macroeconomic factors
    from the requested countries and dates
    """
    data = wbdata.get_dataframe(indicators=indicators,
                                data_date=data_date,
                                convert_date=True,
                                country=countries)
    data = data.unstack(level=0)
    data.columns = ['-'.join(col).strip() for col in data.columns.values]
    print(data.shape, 'Shape with null columns')
    data = data.loc[:, (data.sum(axis=0) != 0)]
    data.dropna(axis='columns', how='all', inplace=True)
    print(data.shape, 'Shape without null columns')
    return data


if __name__ == "__main__":
    data = get_wbdataframe(INDICATORS, DATA_DATE, COUNTRIES)
    data.to_csv('wb_data.csv', index=True)
