import numpy as np
import pandas as pd
import math 


import pandas_datareader as web

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from datetime import date

import matplotlib.pyplot as plt

from keras.models import load_model


scaler = MinMaxScaler(feature_range=(0,1))


df_tomorrow = web.DataReader('KRDMD.IS', data_source='yahoo', start='2019-11-01', end='2022-05-04')


new_df = df_tomorrow.filter(['Close','Low','Volume','Open','High'])

#scale etmem için tansform kullanmam gerekiyordu onun içinde fit etmen gerek dediği için
#new df scale da fit ettim transformu
new_df_scale = new_df.values
last_60_days = scaler.fit_transform(new_df_scale)
last_60_days = new_df[-60:].values

last_60_days_scaled = scaler.transform(last_60_days)

X_test = []

X_test.append(last_60_days_scaled)

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
model = load_model('my_model1.h5')
pred_price_tomorrow = model.predict(X_test)

prediction_copies = np.repeat(pred_price_tomorrow, new_df.shape[1], axis=-1)
pred_price_tomorrow = scaler.inverse_transform(prediction_copies)[:,0]


