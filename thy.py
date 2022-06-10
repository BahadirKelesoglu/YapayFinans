import numpy as np
import pandas as pd
import math 


import pandas_datareader as web

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from datetime import date

import matplotlib.pyplot as plt


# #---------------------------------SALİH----------------------------------------------
# import tweepy as tw
# import pandas as pd
# import datetime
# import time 
# import GetOldTweets3 as got
# import re
# from textblob import TextBlob
# import pandas_datareader as web
# from datetime import date

# df_aapl_close = web.DataReader('AAPL', data_source='yahoo', start='2021-11-01', end=date.today())

# df = pd.read_csv('1_tweets.csv')
# df_tweets_sorted=df.sort_values(by='date')
# df_tweets_sorted.dropna(subset=['Unnamed: 0'],inplace=True)
# df_tweets_sorted['date'] = pd.to_datetime(df_tweets_sorted['date']).dt.date
# df_tweets_sorted.drop("Unnamed: 0", axis=1, inplace=True)
# df_tweets_sorted.drop("tweet_id", axis=1, inplace=True)


# def cleanTxt(text):
#     text = re.sub(r'@[A-Za-z0-9]+', '', text) 
#     text = re.sub(r'#', '', text)
#     text= re.sub(r'RT[\s]+:','',text)
#     text= re.sub(r'https?://\S+','',text)
#     return text

# df_tweets_sorted['tweet']=df_tweets_sorted['tweet'].apply(cleanTxt)

# def getSubjectivity(text):
#     return TextBlob(text).sentiment.subjectivity

# #datanın -1 ile 1 arasında ki cümle pozitifliği
# def getPolarity(text):
#     return TextBlob(text).sentiment.polarity

# #colon olarak gösterme

# df_tweets_sorted['Polarity'] = df_tweets_sorted['tweet'].apply(getPolarity) 

# def getAnalysis(score):
#     if score<0.05:
#         return '0'
#     else:
#         return '1'

# group = df_tweets_sorted["Polarity"].groupby(df_tweets_sorted["date"])
# tweets_polarity=group.mean()

# tweets_polarity = tweets_polarity.to_frame("Polarity")
# tweets_polarity['Label'] = tweets_polarity['Polarity'].apply(getAnalysis)


# # df_aapl_close = df_aapl_close.set_index('Date')
# df_tweets_sorted = df_tweets_sorted.set_index('date')



# s = pd.bdate_range(start="2021/11/01", end=date.today(), freq="C", weekmask="Mon Tue Wed Thu Fri")
# s = s.to_frame()
# s= s.reset_index()
# s['index'] = pd.to_datetime(s['index'], format='%Y%m').dt.date
# s.drop(0, axis=1, inplace=True)



# s= s.reset_index(inplace=False)
# s = s.set_index('index')
# #s = s.reset_index()
# s.index.names= ['date']
# label = pd.merge(s,tweets_polarity, how = 'inner', on = 'date')

# label.drop('level_0', axis=1, inplace=True)
# label.drop('Polarity', axis=1, inplace=True)





# #---------------------------------------END--------------------------------------------




#çizim stili
plt.style.use('dark_background')

#take data 
df = web.DataReader('THYAO.IS', data_source='yahoo', start='2019-11-01', end='2022-05-05')

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
#data['label']=label

#convert the dataframe to a numpy array
dataset = data.values

#get the number of rows to train the model on
#2465 verinin yuzde80 ini aldık .8 ile carparak bunları traine vercez

training_data_len = math.ceil( len(dataset) * .8)

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




#x_train.shape[1] = timesteps
#units, columns 5
model.add(LSTM(500, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))

model.add(LSTM(500))




#feature number
model.add(Dense(1))


#compile the model

model.compile(optimizer='adam', loss='mean_squared_error')




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
model.save('THY.h5')

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()



#creating the testing dataset
#create a new array containing scaled values 1543(1916) to 2003(2469)

test_data = scaled_data[training_data_len - 60: , :]

#Create the data sets x_test and y_test
#seperate close from dataframe for inspect pred and real values in same dataframe
dataset_y_test = df.filter(['Close'])
dataset_y_test = dataset_y_test.values

x_test =[]
y_test = dataset_y_test[training_data_len:, :]


for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0:test_data.shape[1]])
    


  
#convert data to numpy array
x_test = np.array(x_test)


#reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))




#get the models predicted price values

predictions1 = model.predict(x_test)
#unscale etmeden önce x_test de kaç kolon varsa predictionuda kolonları kopyalayarak o boyuta
#çıkarıyoruz çünkü ilk scale da o kadar kolon vardı unscale önceside o kadar kolon olmalı
#0-1 arasına scale ettiğimiz veriyi unscale ediyoruz
prediction_copies = np.repeat(predictions1, train_data.shape[1], axis=-1)
predictions = scaler.inverse_transform(prediction_copies)[:,0]

#şimdi modelimizi RMSE denen yöntemle tahmin kesinliğini değerlendiricez artık fazlalık dataların standart sapmasını alarak yapacağız
mape = np.mean(np.abs((y_test - predictions)/y_test))*100
rmse = np.sqrt(np.mean(predictions - y_test)**2)

#datanın grafiğini çiz
train =data[0:training_data_len]
data_valid = data.filter(['Close'])
valid = data_valid[training_data_len:]

#valide kolon ekledik
valid['Tahmin'] = predictions

#visualize the data

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Gün')
plt.ylabel('Kapanış $', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Tahmin']])
plt.legend(['train', 'Validate', 'predictions'], loc='lower right')
plt.show()