### Description : Creating a streamlit interface for stock price forecast

## importing libraries

from os import error
from re import template
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
from statsmodels.tsa import arima, seasonal
plt.style.use('ggplot')
import plotly.graph_objs as go
import yfinance as yf
import pandas_market_calendars as mcal
import itertools
import warnings as wr
wr.filterwarnings("ignore")
import streamlit as st

from sklearn import metrics
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

### Titling

st.title(
    """Stock Price Forecast"""
)
scrip_name = st.text_input("Enter SCRIP Name")
st.write('for eg. Wipro as WIPRO ( Ticker Name )')

## Downloading data
data = yf.download(tickers = scrip_name + '.NS', period = '240d', interval='1d')
ploting_data = data.reset_index().copy()
data = data.reset_index()
df = data[['Close', 'Date']]

## Generating Time Stamps

bse = mcal.get_calendar('BSE')
early = bse.schedule(start_date = '2020-11-25', end_date = '2021-11-15')
time_stamp = mcal.date_range(early, frequency='1D')

df['Time_Stamp'] = time_stamp.date
df.set_index('Time_Stamp', inplace = True)
df.drop('Date', axis = 1, inplace = True)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def main(scrip_name, df):
    
    ## Generating Training and Testing for user validation

    train = df[0:int(len(df)*0.80)]
    test = df[int(len(df)*0.80): ]

    ## Triple Exponential Smoothing

    ### Triple Exponential Smoothing

    def tes_model(train, test, col, seasonal):
        
        TES_train = train.copy()
        TES_test = test.copy()
        model_TES = ExponentialSmoothing(TES_train[col],trend='additive',seasonal=seasonal, seasonal_periods=6)
        
        resultsDf_8_2 = pd.DataFrame({'Alpha Values':[],'Beta Values':[],'Gamma Values':[],'Train RMSE':[],'Test RMSE': []})
        
        for i in np.arange(0.3,1.1,0.1):
            for j in np.arange(0.3,1.1,0.1):
                for k in np.arange(0.3,1.1,0.1):
                    model_TES_alpha_i_j_k = model_TES.fit(smoothing_level=i,smoothing_trend=j,smoothing_seasonal=k,optimized=False,use_brute=True)
                    TES_train['predict',i,j,k] = model_TES_alpha_i_j_k.fittedvalues
                    TES_test['predict',i,j,k] = model_TES_alpha_i_j_k.forecast(steps = len(TES_test)).values
            
                    rmse_model8_train = metrics.mean_squared_error(TES_train[col],TES_train['predict',i,j,k],squared=False)
                    rmse_model8_test = metrics.mean_squared_error(TES_test[col],TES_test['predict',i,j,k],squared=False)
                    
                    mape_train = metrics.mean_absolute_percentage_error(TES_train[col], TES_train['predict', i,j,k]) * 100
                    mape_test = metrics.mean_absolute_percentage_error(TES_test[col], TES_test['predict', i,j,k]) * 100
                    
                    
                    resultsDf_8_2 = resultsDf_8_2.append({'Alpha Values':i,'Beta Values':j,'Gamma Values':k,
                                                        'Train RMSE':rmse_model8_train,'Test RMSE':rmse_model8_test,
                                                        'Train MAPE' : mape_train, 'Test MAPE': mape_test}
                                                        , ignore_index=True)
                    
        return resultsDf_8_2


    result_tes_add = tes_model(train, test, 'Close', 'additive')
    result_tes_mul = tes_model(train, test, 'Close', 'multiplicative')

    ## For Additive 
    tes_add_sorted = result_tes_add.sort_values('Test RMSE', ascending=True).head()
    index_value_add = tes_add_sorted.index[0] 
    alpha_add = tes_add_sorted['Alpha Values'][index_value_add]
    beta_add = tes_add_sorted['Beta Values'][index_value_add]
    gamma_add = tes_add_sorted['Gamma Values'][index_value_add]

    ## For Multiplicative

    tes_mul_sorted = result_tes_mul.sort_values('Test RMSE', ascending=True).head()
    index_value_mul = tes_mul_sorted.index[0]
    alpha_mul = tes_mul_sorted['Alpha Values'][index_value_mul]
    beta_mul = tes_mul_sorted['Beta Values'][index_value_mul]
    gamma_mul = tes_mul_sorted['Gamma Values'][index_value_mul]


    ### TES main Function
    ### Double Exponential Main Function
    def tes_main(train, test, col, alpha, beta, gamma, seasonal):
        TES_train = train.copy()
        TES_test = test.copy()

        model_TES = ExponentialSmoothing(TES_train[col], trend = 'additive', seasonal = seasonal, seasonal_periods=6)
        model_TES_autofit = model_TES.fit(smoothing_level=alpha,smoothing_trend = beta, smoothing_seasonal = gamma)
        print(model_TES_autofit.params)
        
        ## Plotting on both the Training and Test data
        TES_test['predict'] = model_TES_autofit.forecast(len(TES_test)).values
        
        rmse = metrics.mean_squared_error(TES_test[col], TES_test['predict'], squared = False)
        mape = metrics.mean_absolute_percentage_error(TES_test[col], TES_test['predict']) * 100
        
        return rmse, mape

    rmse_add, mape_add  = tes_main(train, test, 'Close', alpha_add, beta_add, gamma_add, 'additive')
    rmse_mul, mape_mul  = tes_main(train, test, 'Close', alpha_mul, beta_mul, gamma_mul, 'multiplicative')

    print(rmse_add, mape_add, rmse_mul, mape_mul)

    ## Choosing best model 

    if rmse_add > rmse_mul:
        print('Additive')
        TES_model_final = ExponentialSmoothing(df['Close'], trend = 'additive', seasonal = 'additive', seasonal_periods=6)
        TES_model_final_auto = TES_model_final.fit(smoothing_level = alpha_add, smoothing_trend = beta_add, smoothing_seasonal = gamma_add)
        forecast_tes = TES_model_final_auto.forecast(steps = 12).values
        rmse_holt = rmse_add.copy()
    elif rmse_add < rmse_mul:
        print('Multiplicative')
        TES_model_final = ExponentialSmoothing(df['Close'], trend = 'additive', seasonal = 'multiplicative', seasonal_periods=6)
        TES_model_final_auto = TES_model_final.fit(smoothing_level = alpha_mul, smoothing_trend = beta_mul, smoothing_seasonal = gamma_mul)
        forecast_tes = TES_model_final_auto.forecast(steps = 12).values
        rmse_holt = rmse_mul.copy()

    ### ARIMA Model

    p = q = range(0,3)
    d = range(1,2)
    pdq = list(itertools.product(p,d,q))

    ARIMA_AIC = pd.DataFrame(columns =  ['param', 'AIC'])

    for param in pdq:
        ARIMA_model = ARIMA(train['Close'].values, order = param).fit()
        ARIMA_AIC = ARIMA_AIC.append({'param' : param, 'AIC' : ARIMA_model.aic}, ignore_index = True)

    arima_sorted = ARIMA_AIC.sort_values('AIC', ascending = True).head(5)
    index_value_arima = arima_sorted.index[0]
    order_arima = arima_sorted['param'][index_value_arima]

    ARIMA_model = ARIMA(train['Close'].values, order  = order_arima).fit()
    ARIMA_predict = ARIMA_model.forecast(len(test))

    arima_rmse = metrics.mean_squared_error(test, ARIMA_predict[0], squared = False)

    ARIMA_final = ARIMA(df['Close'].values, order = order_arima).fit()
    forecast_arima = ARIMA_final.forecast(steps = 12)

    ### SARIMA Model

    p = q = range(0,3)
    d = range(0,2)
    D = range(0,2)
    pdq = list(itertools.product(p,d,q))
    model_pdq = [(x[0], x[1], x[2], 6) for x in list(itertools.product(p, D, q))]


    SARIMA_AIC = pd.DataFrame(columns=['param','seasonal', 'AIC'])

    for param in pdq:
        for param_seasonal in model_pdq:
            SARIMA_model = SARIMAX(train['Close'].values,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                
            results_SARIMA = SARIMA_model.fit(maxiter=1000)
            print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results_SARIMA.aic))
            SARIMA_AIC = SARIMA_AIC.append({'param':param,'seasonal':param_seasonal ,'AIC': results_SARIMA.aic}, ignore_index=True)


    sarima_sorted = SARIMA_AIC.sort_values('AIC', ascending = True).head()
    index_value_sarima = sarima_sorted.index[0]
    order_sarima = sarima_sorted['param'][index_value_sarima]
    seasonal_order_sarima = sarima_sorted['seasonal'][index_value_sarima]

    SARIMA_model = SARIMAX(train['Close'], order = order_sarima, seasonal_order=seasonal_order_sarima,
                            enforce_stationarity=False, enforce_invertibility=False).fit()
    SARIMA_predict = SARIMA_model.forecast(len(test)).values

    sarima_rmse = metrics.mean_squared_error(test, SARIMA_predict, squared = False)


    SARIMA_final = SARIMAX(df['Close'], order = order_sarima, seasonal_order = seasonal_order_sarima, 
                            enforce_invertibility=False, enforce_stationarity=False).fit()
    forecast_sarima = SARIMA_final.forecast(steps = 12).values



    forecasting_range = pd.date_range(start = df.index[-1], periods = 12, freq = BDay()).date
    forecast_table = pd.DataFrame({'Holt-Winter' : forecast_tes, 'ARIMA' : forecast_arima[0], 'SARIMA-6' : forecast_sarima},
    index = forecasting_range)

    ### Accuracy Tables:
    rmse_series = [rmse_holt, arima_rmse, sarima_rmse]
    error_table = pd.DataFrame(rmse_series,columns = ['RMSE'] ,index = ['Holt Winter', 'ARIMA', 'Seasonal ARIMA-6'])

    return error_table, forecast_table

    
if st.button("Submit"):

    ### Plotly Graph for displaying stock price over 365 days

    fig = go.Figure(data = [go.Candlestick(x = ploting_data['Date'],
                    open = ploting_data['Open'], high = ploting_data['High'], low = ploting_data['Low'], close = ploting_data['Close'])])
    fig.update_layout(xaxis_rangeslider_visible = False, template = 'plotly_dark',
                        width = 800, height = 500)
    st.write("Stock Price Chart (1 year)")
    st.plotly_chart(fig)

    error_table, forecast_table = main(scrip_name, df)
    st.write("Forecasted Price for 12 days")
    st.dataframe(forecast_table)
    st.write(" ")
    st.write(" ")
    st.sidebar.write("For Choosing the Best Forecast refer to the table below")
    st.sidebar.write("The Model with the best forecast will have the lowest score")
    st.sidebar.dataframe(error_table)


