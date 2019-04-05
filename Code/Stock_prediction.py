# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:32:15 2019

@author: t4nis
"""
from pmdarima.arima import auto_arima
from pyramid.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from plotly.plotly import plot_mpl
import plotly
from matplotlib.pyplot import figure
from sklearn.preprocessing import MinMaxScaler
plotly.tools.set_credentials_file(username='t4nish', api_key='0Qi04rOBqP4wELkOdZyd')
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


df = pd.read_csv('C:/Users/t4nis/Desktop/Project/SP_Daily.csv')
df.head()

scaler = MinMaxScaler(feature_range=(0, 1))

#setting index as date
#convert to daterange
df['Date'] = pd.to_datetime(df.Date,format='%Y/%m/%d')
df.index = df['Date']
data = df.sort_index(ascending=True, axis=0)
data.index
freq=data.asfreq('D')
freq['Close_fill'] = data['Close'].asfreq('D', method='ffill')
freq=freq['Close_fill']
df_size = len(freq)
#freq.fillna((freq.mean()),inplace=True)


#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')

train_size = round(df_size*90/100)
test_size = round(df_size*10/100)
df_train = freq[:train_size]
df_test = freq[-test_size:]
training = df_train['Close']
validation = df_test['Close']
￼


#Check for Trend and Seasonality
result = seasonal_decompose(data.Close, model='multiplicative', freq=365)
fig = result.plot()
plt.plot(fig)




#ARIMA 
model = auto_arima(freq, start_p=0, start_q=0,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)
print(model.aic())
#Fit ARIMA: order=(2, 1, 3) seasonal_order=(0, 1, 1, 12); AIC=84321.748, BIC=84380.180, Fit time=141.175 secondsmodel.fit(df_train)

forecast = model_fit.predict(n_periods=(len(df_test)))
forecast = pd.DataFrame(forecast,index = df_test.index,columns=['Prediction'])

rms=np.sqrt(np.mean(np.power((np.array(df_test)-np.array(forecast['Prediction'])),2)))
rms
plt.figure(figsize=(14,8))
plt.plot(df_train)
plt.plot(df_test)
plt.plot(forecast['Prediction'])


model=SARIMAX(df_train,order=(2,1,3),seasonal_order=(0,1,1,12))
model_fit=model.fit()
print(model_fit.summary())
forecast=model_fit.predict(start='2016-02-06 00:00:00', end='2019-02-08 00:00:00')
pd.date_range(end='1/1/2018', periods=8) 
plt.plot(df_train)
plt.plot(df_test)
plt.plot(forecast)



#Exponential Smoothening
fit1 = SimpleExpSmoothing(df_train).fit()
fcast1 = fit1.forecast(len(df_test))
forecast_1 = pd.DataFrame(fcast1,index = df_test.index,columns=['Prediction'])

plt.plot(df_train['Close'])
plt.plot(df_test['Close'])
plt.plot(forecast_1['Prediction'])


y_hat_avg = validation.copy()
fit2 = SimpleExpSmoothing(training).fit(smoothing_level=0.2,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(validation))
plt.figure(figsize=(16,8))
plt.plot(training, label='Train')
plt.plot(validation, label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()
