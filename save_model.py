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

#çizim stili
plt.style.use('dark_background')

#take data 
df = web.DataReader('ASELS.IS', data_source='yahoo', start='2019-11-01', end=date.today())

#visualize the closing price history

plt.figure(figsize=(16,8))
#grafik başlığı
plt.title('Kapanis')
#close kolonunu al
plt.plot(df['Close'])
plt.xlabel('Zaman', fontsize=20)
plt.ylabel('kapanis fiyati ($)', fontsize=20)
plt.show()

#create a new dataframe with only the close column

data = df.filter(['Close','Low','Volume','Open','High'])

#convert the dataframe to a numpy array
dataset = data.values


#get the number of rows to train the model on
#2465 verinin yuzde80 ini aldık .8 ile carparak bunları traine vercez

training_data_len = len(dataset)

#scale data
#close verilerimizi 1 ve 0 arasında olacak şekilde ölcekledik
#sebebi muhtemelen daha rahat işlem yapabilmemiz için

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


#create training data set
train_data = scaled_data[0:training_data_len,:]


#Split the data into x_train and y_train data sets
x_train=[]
y_train = []

#x train ilk 60 günler veri olucak bağımsız değişkenler
#y train 61. gün olarak predict etceğimiz bağımlı değişken
#foru dikkatli incele ilk 60 günden sonraki her günü her bir forla geçmiş 60 günle karşılaştırıyor
for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0:train_data.shape[1]])
        y_train.append(train_data[i,0])

# #we will change them and i keep them to look at their first situation
first_x_train = x_train
first_y_train = y_train

# #convert x_train and y_train to numpy arrays
# #so we can use them for lstm model
x_train = np.array(x_train)
y_train = np.array(y_train)

#reshape the data
#our train data's are 2 dimensional
#lstm networks expect the input 3 dimensional
#1- number of samples
#2- number of time steps
#3- number of features

#np.reshape(x_train, (number of rows, number of column, features=closingpriceso=1))

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

#build the lstm model architecture
#choose model
model = Sequential()
#we will add some layer idk spesificly
#LSTM(neurons, return?, input_shape=(columns, features))





model.add(LSTM(500, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(500))
model.add(Dense(1))


#compile the model

model.compile(optimizer='adam', loss='mse')




#x = model.summary()

#validation loss relevant with epoch
#training loss relevant with batch size
# Overfitting if: training loss << validation loss

# Underfitting if: training loss >> validation loss

# Just right if training loss ~ validation loss

# Reasons behind overfitting:
# Using a complex model for a simple problem which picks up the noise from the data. Example: Fitting a neural network to the Iris dataset.
# Small datasets, as the training set may not be a right representation of the universe.

# Reasons behind underfitting:
# Using a simple model for a complex problem which doesn’t learn all the patterns in the data. Example: Using a logistic regression for image classification
# The underlying data has no inherent pattern. Example, trying to predict a student’s marks with his father’s weight.

#train the model
history = model.fit(x_train, y_train, batch_size=32, validation_split=0.1, epochs=50)




model.save('my_model1.h5')

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
