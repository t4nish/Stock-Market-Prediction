# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:32:15 2019

@author: t4nis
"""

from pyramid.arima import auto_arima
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
df['Date'] = pd.to_datetime(df.Date,format='%Y/%m/%d')
df.index = df['Date']

#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')


data = df.sort_index(ascending=True, axis=0)
df_size = len(data)
train_size = round(df_size*90/100)
test_size = round(df_size*10/100)
df_model=data.Close
df_train = data[:train_size]
df_test = data[-test_size:]
￼


#Check for Trend and Seasonality
result = seasonal_decompose(data.Close, model='multiplicative', freq=365)
fig = result.plot()
plt.plot(fig)


training = df_train['Close']
validation = df_test['Close']

#ARIMA 
model = auto_arima(df_model, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)
print(model.aic())
#Fit ARIMA: order=(3, 1, 3) seasonal_order=(0, 1, 1, 12); AIC=60963.421, BIC=61025.812, Fit time=132.644 seconds
model.fit(training)

forecast = model.predict(n_periods=(len(validation)))
forecast = pd.DataFrame(forecast,index = df_test.index,columns=['Prediction'])

rms=np.sqrt(np.mean(np.power((np.array(df_test['Close'])-np.array(forecast['Prediction'])),2)))
rms
plt.figure(figsize=(16,8))
plt.plot(df_train['Close'])
plt.plot(df_test['Close'])
plt.plot(forecast['Prediction'])

#Exponential Smoothening
fit1 = SimpleExpSmoothing(training).fit()
fcast1 = fit1.forecast(len(validation))
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
