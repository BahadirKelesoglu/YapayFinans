

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time


plt.style.use('dark_background')

import pandas_datareader as web
from datetime import date
import datetime
from statsmodels.tsa.arima_model import ARIMA

df = web.DataReader('THYAO.IS', data_source='yahoo', start='2019-11-01', end=date.today())
df=df[["Close"]].copy()
describe=df.describe()

n= int(len(df) * 0.848)
train = df.Close[:n]
test = df.Close[n:]
x_train=[]
x_train = (df.Close[:n])
#kardemir, KOC 3-2-1 yaptım
model=ARIMA(train,order=(5,2,1))
result=model.fit(disp=0)

#tomorrow days prediction
step= 1
fc, se, conf = result.forecast(step)
fc=pd.Series(fc)
real_result = fc
result.summary()
for i in range(1,len(test)):
        x_train = (df.Close[:n + i])
        print('length of x_train = ', len(x_train))
        model=ARIMA(x_train,order=(5,2,1))
        result=model.fit(disp=0)
        #tomorrow days prediction
        print(i)
        step= 1
        fc, se, conf = result.forecast(step)
        fc=pd.Series(fc)
        print('fc=', fc)
        real_result = real_result.append(fc)
        
        





real_result.index=test.index

rmse = np.sqrt(np.mean(real_result - test)**2)

plt.figure(figsize=(12,6))
plt.plot(test, label="actual", c='tab:blue')
plt.plot(real_result, label="forecast", c='tab:red')
plt.title("Forecast vs Actual")
plt.legend(loc="upper left")
plt.savefig('THY_tsa.png')

