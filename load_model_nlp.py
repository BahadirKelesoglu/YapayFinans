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

#---------------------------------SALİH----------------------------------------------
import tweepy as tw
import datetime
import time 
import GetOldTweets3 as got
import re
from textblob import TextBlob
import pandas_datareader as web
from datetime import date

df_aapl_close = web.DataReader('THYAO.IS', data_source='yahoo', start='2019-11-01', end='2022-05-05')

df = pd.read_csv('genel_tweet.csv')
df_tweets_sorted=df.sort_values(by='date')
df_tweets_sorted.dropna(subset=['Unnamed: 0'],inplace=True)
df_tweets_sorted['date'] = pd.to_datetime(df_tweets_sorted['date']).dt.date
df_tweets_sorted.drop("Unnamed: 0", axis=1, inplace=True)
df_tweets_sorted.drop("tweet_id", axis=1, inplace=True)


def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) 
    text = re.sub(r'#', '', text)
    text= re.sub(r'RT[\s]+:','',text)
    text= re.sub(r'https?://\S+','',text)
    return text

df_tweets_sorted['tweet']=df_tweets_sorted['tweet'].apply(cleanTxt)

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#datanın -1 ile 1 arasında ki cümle pozitifliği
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

#colon olarak gösterme

df_tweets_sorted['Polarity'] = df_tweets_sorted['tweet'].apply(getPolarity) 

def getAnalysis(score):
    if score<0.05:
        return '0'
    else:
        return '1'

group = df_tweets_sorted["Polarity"].groupby(df_tweets_sorted["date"])
tweets_polarity=group.mean()

tweets_polarity = tweets_polarity.to_frame("Polarity")
tweets_polarity['Label'] = tweets_polarity['Polarity'].apply(getAnalysis)


# df_aapl_close = df_aapl_close.set_index('Date')
df_tweets_sorted = df_tweets_sorted.set_index('date')



s = pd.bdate_range(start="2019/11/01", end=date.today(), freq="C", weekmask="Mon Tue Wed Thu Fri")
s = s.to_frame()
s= s.reset_index()
s['index'] = pd.to_datetime(s['index'], format='%Y%m').dt.date
s.drop(0, axis=1, inplace=True)



s= s.reset_index(inplace=False)
s = s.set_index('index')
#s = s.reset_index()
s.index.names= ['date']
label = pd.merge(s,tweets_polarity, how = 'inner', on = 'date')

label.drop('level_0', axis=1, inplace=True)
label.drop('Polarity', axis=1, inplace=True)





#---------------------------------------END--------------------------------------------


scaler = MinMaxScaler(feature_range=(0,1))


df_tomorrow = web.DataReader('THYAO.IS', data_source='yahoo', start='2019-11-01', end='2022-05-04')
df_tomorrow['Label'] = label['Label']

new_df = df_tomorrow.filter(['Close','Low','Volume','Open','High','Label'])

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
model = load_model('THY_WITH_NLP.h5')
pred_price_tomorrow = model.predict(X_test)

prediction_copies = np.repeat(pred_price_tomorrow, new_df.shape[1], axis=-1)
pred_price_tomorrow = scaler.inverse_transform(prediction_copies)[:,0]



