import webbrowser
import re
import sys
import os
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
import tkinter.filedialog
from pathlib import Path
from PIL import Image, ImageTk
choose = 'Select Your Product'
choose_algorithm = 'Select An Algorithm'




sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from tkdesigner.designer import Designer
except ModuleNotFoundError:
    raise RuntimeError("Couldn't add tkdesigner to the PATH.")


# Path to asset files for this GUI window.
ASSETS_PATH = Path(__file__).resolve().parent / "assets"

# Required in order to add data files to Windows executable
path = getattr(sys, '_MEIPASS', os.getcwd())
os.chdir(path)

output_path = ""

window = tk.Tk()
logo = tk.PhotoImage(file=ASSETS_PATH / "iconbitmap.gif")
window.call('wm', 'iconphoto', window._w, logo)
window.title("Yapay Finans")
window.geometry("1200x700")
window.configure(bg="#3A7FF6")
canvas = tk.Canvas(
    #background sol tarafın burası
    window, bg="#292b4d", height=700, width=1200,
    bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)
canvas.create_rectangle(800, 0, 800 + 800, 0 + 700, fill="#FCFCFC", outline="")
canvas.create_rectangle(100, 280, 100 + 100, 280 + 8, fill="black", outline="")

global predict_result


def tsa_tomorrow(product):
    from IPython.core.debugger import set_trace

    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import time
    
    plt.style.use('dark_background')
    
    import pandas_datareader as web
    from datetime import date
    import datetime
    
    df = web.DataReader(product, data_source='yahoo', start='2019-11-01', end=date.today())
    df=df[["Close"]].copy()
    describe=df.describe()
    
    n= int(len(df))
    train = df.Close[:n]
    
    
    from statsmodels.tsa.arima_model import ARIMA
    #7,1,2
    model=ARIMA(train,order=(0,2,1))
    result=model.fit(disp=0)
    
    print(result.summary())
    #1 days prediction
    step=1
    fc, se, conf = result.forecast(step)
    
    
    
    
    fc=pd.Series(fc)
   
    
    
    valid=pd.date_range(datetime.date.today() + datetime.timedelta(days=1), periods=step, freq='B')
    valid=valid.to_frame()
    fc = pd.DataFrame(fc, columns = ['Prediction'])
    
    valid=valid.set_index(0)
    
    valid['Prediction']=fc['Prediction'].values
    fc=valid
    valid1 = valid
    valid1 = pd.DataFrame(valid1, columns=['Prediction'])
    valid1['Prediction'] = valid['Prediction']
    global tsa_result
    tsa_result=valid1.values
    

def tsa_plot(save_plot_name,product, test_len):
    from IPython.core.debugger import set_trace

    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import time
    
    plt.style.use('dark_background')
    
    import pandas_datareader as web
    from datetime import date
    import datetime
    
    df = web.DataReader(product, data_source='yahoo', start='2019-11-01', end=date.today())
    df=df[["Close"]].copy()
    describe=df.describe()
    
    n= int(len(df) * test_len)
    train = df.Close[:n]
    test = df.Close[n:]
    
    
    from statsmodels.tsa.arima_model import ARIMA
    #7,1,2
    model=ARIMA(train,order=(5,2,1))
    result=model.fit(disp=0)
    
    print(result.summary())
    #6 days prediction
    step= int(len(test))
    fc, se, conf = result.forecast(step)
    
    
    
    
    fc=pd.Series(fc, index= test[:step].index)
    
    lower = pd.Series(conf[:, 0], index =test[:step].index)
    
    upper = pd.Series(conf[:, 1], index=test[:step].index)
    
    plt.figure(figsize=(12,6))
    plt.plot(test[:step], label="actual", c='tab:blue')
    plt.plot(fc, label="forecast", c='tab:red')
    plt.fill_between(lower.index, lower, upper, color="k", alpha=0.1)
    plt.title("Forecast vs Actual")
    plt.legend(loc="upper left")
    plt.savefig(save_plot_name)
    
    valid=pd.date_range(("2021-08-13"), periods=step, freq='C')
    valid=valid.to_frame()
    fc = pd.DataFrame(fc, columns = ['Prediction'])
    
    valid=valid.set_index(0)
    
    valid['Prediction']=fc['Prediction'].values
    fc=valid
    global rmse_tsa
    rmse_tsa = np.sqrt(np.mean(fc - test)**2)
    
    valid1 = valid
    valid1 = pd.DataFrame(valid1, columns=['Prediction'])
    valid1['Prediction'] = valid['Prediction']

def Show_Plot_NLP(product, model,save_plot_name):
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
  import pandas as pd
  import datetime
  import time 
  import GetOldTweets3 as got
  import re
  from textblob import TextBlob
  import pandas_datareader as web
  from datetime import date

  df_aapl_close = web.DataReader(product, data_source='yahoo', start='2019-11-01', end='2022-06-08')

  df = pd.read_csv('deneme.csv')
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




  #çizim stili
  plt.style.use('dark_background')

  #take data 
  df = web.DataReader(product, data_source='yahoo', start='2019-11-01', end='2022-06-08')
  df['Label'] = label['Label']
  #visualize the closing price history

  #create a new dataframe with only the close column

  data = df.filter(['Close','Low','Volume','Open','High', 'Label'])
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
  model = load_model(model)
  
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
  global rmse_nlp
  rmse_nlp = np.sqrt(np.mean(predictions - y_test)**2)

  #datanın grafiğini çiz
  train =data[0:training_data_len]
  data_valid = data.filter(['Close'])
  valid = data_valid[training_data_len:]

  #valide kolon ekledik
  valid['Tahmin'] = predictions

  #visualize the data

  plt.figure(figsize=(12,6))
  plt.ylabel('Kapanış $', fontsize=25)
  plt.plot(valid[['Close']], c='tab:blue' )
  plt.plot(valid[['Tahmin']], c='tab:red' )
  plt.legend(['Validate', 'predictions'], loc='lower right')
  plt.xticks(rotation=45)
  plt.savefig(save_plot_name)
    
def load_with_nlp(product, model):
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

    df_aapl_close = web.DataReader(product, data_source='yahoo', start='2019-11-01', end='2022-06-08')

    df = pd.read_csv('deneme.csv')
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


    df_tomorrow = web.DataReader(product, data_source='yahoo', start='2019-11-01', end='2022-06-08')
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
    model = load_model(model)
    pred_price_tomorrow = model.predict(X_test)

    prediction_copies = np.repeat(pred_price_tomorrow, new_df.shape[1], axis=-1)
    pred_price_tomorrow = scaler.inverse_transform(prediction_copies)[:,0]
    global result_load_nlp
    result_load_nlp=pred_price_tomorrow
    

def load(product, model):
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
  
    df_tomorrow = web.DataReader(product, data_source='yahoo', start='2019-11-01', end=date.today())
    
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
    model = load_model(model)
    pred_price_tomorrow = model.predict(X_test)

    prediction_copies = np.repeat(pred_price_tomorrow, new_df.shape[1], axis=-1)
    pred_price_tomorrow = scaler.inverse_transform(prediction_copies)[:,0]
    global result_load
    result_load=pred_price_tomorrow
                                      
                                      
#%% Show_Plot_Wıthout_NLP
def Show_Plot(product, model,save_plot_name):
    import pandas_datareader as web
    
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout
    from datetime import date
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.models import load_model

    #çizim stili
    plt.style.use('dark_background')
    
    #take data 
    df = web.DataReader(product, data_source='yahoo', start='2019-11-01', end=date.today())
    
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
    
    model = load_model(model)
    
    
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
    global rmse
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    
    
    #datanın grafiğini çiz
    train =data[0:training_data_len]
    data_valid = data.filter(['Close'])
    valid = data_valid[training_data_len:]
    
    #valide kolon ekledik
    valid['Tahmin'] = predictions
    
    #visualize the data
    
    plt.figure(figsize=(12,6))
    plt.ylabel('Kapanış $', fontsize=25)
    plt.plot(valid[['Close']], c='tab:blue' )
    plt.plot(valid[['Tahmin']], c='tab:red' )
    plt.legend(['Validate', 'predictions'], loc='lower right')
    plt.xticks(rotation=45)
    plt.savefig(save_plot_name)


#------------------------------------------------------------END--------------------------------
#%%

def show_plot(): #variable
    print(choose) 
    
    
    if(choose=='ASELSAN') & (choose_algorithm == 'LSTM'):
    
        # sets the geometry of toplevel
        Show_Plot('ASELS.IS', 'aselsan.h5', 'aselsan.png')
        img_old=Image.open('aselsan.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text=round(rmse,3), bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
        
        
        
    
    elif(choose=='ASELSAN') & (choose_algorithm == 'LSTM And NLP'):
        
        Show_Plot_NLP('ASELS.IS', 'ASELSAN_WITH_NLP.h5', 'aselsan_with_nlp.png')
        img_old=Image.open('aselsan_with_nlp.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text=round(rmse_nlp,3), bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
        
    
    elif(choose=='KARDEMIR') & (choose_algorithm == 'LSTM') :
      
        Show_Plot('KRDMD.IS', 'kardemir.h5', 'kardemir.png')
        img_old=Image.open('kardemir.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text=round(rmse,3), bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)


    elif(choose=='KARDEMIR') & (choose_algorithm == 'LSTM And NLP') :
       
        Show_Plot_NLP('KRDMD.IS', 'KARDEMIR_WITH_NLP.h5', 'kardemir_with_nlp.png')
        img_old=Image.open('kardemir_with_nlp.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text=round(rmse_nlp,3), bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
                    

    elif(choose=='QNB') & (choose_algorithm == 'LSTM'):
        
        Show_Plot('QNBFB.IS', 'QNB.h5', 'QNB.png')
        img_old=Image.open('QNB.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text=round(rmse,3), bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
        
        
    
    elif(choose=='QNB') & (choose_algorithm == 'LSTM And NLP'):
    
        Show_Plot_NLP('QNBFB.IS', 'QNB_WITH_NLP.h5', 'QNB_with_nlp.png')
        img_old=Image.open('QNB_with_nlp.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text=round(rmse_nlp,3), bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
          
        
        
    elif(choose=='KOC') & (choose_algorithm == 'LSTM'):
       
        Show_Plot('KCHOL.IS', 'KOC.h5', 'KOC.png')
        img_old=Image.open('KOC.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text=round(rmse,3), bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
     
        
        
    elif(choose=='KOC') & (choose_algorithm == 'LSTM And NLP'):
        
        Show_Plot_NLP('KCHOL.IS', 'KOC_WITH_NLP.h5', 'KOC_with_nlp.png')
        img_old=Image.open('KOC_with_nlp.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text=round(rmse_nlp,3), bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
            
        
        
    elif(choose=='BIM') & (choose_algorithm == 'LSTM'):
      
        Show_Plot('BIMAS.IS', 'BIM.h5', 'BIM.png')
        img_old=Image.open('BIM.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text=round(rmse,3), bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
          
        
    
    elif(choose=='BIM') & (choose_algorithm == 'LSTM And NLP'):
        
        Show_Plot_NLP('BIMAS.IS', 'BIM_WITH_NLP.h5', 'BIM_with_nlp.png')
        img_old=Image.open('BIM_with_nlp.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text=round(rmse_nlp,3), bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
        
    
    
    elif(choose=='THY') & (choose_algorithm == 'LSTM And NLP'):
       
        Show_Plot_NLP('THYAO.IS', 'THY_WITH_NLP.h5', 'THY_with_nlp.png')
        img_old=Image.open('THY_with_nlp.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text=round(rmse_nlp,3), bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
      
        
        
    elif(choose=='THY') & (choose_algorithm == 'LSTM'):
      
        Show_Plot('THYAO.IS', 'THY.h5', 'THY.png')
        img_old=Image.open('THY.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text=round(rmse,3), bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
            
def call_product():
    print(choose) 
    if(choose=='ASELSAN') & (choose_algorithm == 'LSTM'):
        product = 'ASELS.IS'
        model = 'aselsan.h5'
        load(product, model)
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=result_load, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
        
    elif(choose=='ASELSAN') & (choose_algorithm == 'LSTM And NLP'):
        product = 'ASELS.IS'
        model = 'ASELSAN_WITH_NLP.h5'
        load_with_nlp(product, model)
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=result_load_nlp, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
    
    elif(choose=='KARDEMIR') & (choose_algorithm == 'LSTM'):
        product = 'KRDMD.IS'
        model = 'kardemir.h5'
        load(product, model)
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=result_load, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
        
    elif(choose=='KARDEMIR') & (choose_algorithm == 'LSTM And NLP'):
        product = 'KRDMD.IS'
        model = 'KARDEMIR_WITH_NLP.h5'
        load_with_nlp(product, model)
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=result_load_nlp, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
    
    elif(choose=='QNB') & (choose_algorithm == 'LSTM'):
        product = 'QNBFB.IS'
        model = 'QNB.h5'
        load(product, model)
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=result_load, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
        
    elif(choose=='QNB') & (choose_algorithm == 'LSTM And NLP'):
        product = 'QNBFB.IS'
        model = 'QNB_WITH_NLP.h5'
        load_with_nlp(product, model)
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=result_load_nlp, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
        
    elif(choose=='KOC') & (choose_algorithm == 'LSTM'):
        product = 'KCHOL.IS'
        model = 'KOC.h5'
        load(product, model)
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=result_load, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
        
    elif(choose=='KOC') & (choose_algorithm == 'LSTM And NLP'):
        product = 'KCHOL.IS'
        model = 'KOC_WITH_NLP.h5'
        load_with_nlp(product, model)
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=result_load_nlp, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
    
    elif(choose=='THY') & (choose_algorithm == 'LSTM'):
        product = 'THYAO.IS'
        model = 'THY.h5'
        load(product, model)
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=result_load, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
        
    elif(choose=='THY') & (choose_algorithm == 'LSTM And NLP'):
        product = 'THYAO.IS'
        model = 'THY_WITH_NLP.h5'
        load_with_nlp(product, model)
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=result_load_nlp, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
    
    elif(choose=='BIM') & (choose_algorithm == 'LSTM'):
        product = 'BIMAS.IS'
        model = 'BIM.h5'
        load(product, model)
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=result_load, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
        
    elif(choose=='BIM') & (choose_algorithm == 'LSTM And NLP'):
        product = 'BIMAS.IS'
        model = 'BIM_WITH_NLP.h5'
        load_with_nlp(product, model)
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=result_load_nlp, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)

def call_tsa():
    print(choose) 
    if(choose=='ASELSAN') & (choose_algorithm == 'TSA'):
        product = 'ASELS.IS'
        tsa_tomorrow(product)
        
        img_old=Image.open('aselsan_tsa.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=tsa_result, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text="0.011", bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
        
    
    elif(choose=='KARDEMIR') & (choose_algorithm == 'TSA'):
            product = 'KRDMD.IS'
            
            tsa_tomorrow(product)
            
            img_old=Image.open('kardemir_tsa.png')
            img_resized = img_old.resize((700,400))
            img = ImageTk.PhotoImage(img_resized)
            title = tk.Label(
                image=img, bg="black",
                fg="white", font=("Arial-BoldMT", int(20.0)))
            title.place(x=-35, y=-35)
            title.image = img
            
            predict_show = tk.Label(
                text="Tomorrow Prediction", bg="black",
                fg="white", font=("Arial-BoldMT", int(15.0)))
            predict_show.place(x=600.0, y=120.0)
            
            predict_result = tk.Label(
                text=tsa_result, bg="black",
                fg="white", font=("Arial-BoldMT", int(10.0)))
            predict_result.place(x=645.0, y=160.0)
            
            rmse_show = tk.Label(
                text="RMSE", bg="black",
                fg="white", font=("Arial-BoldMT", int(15.0)))
            rmse_show.place(x=670.0, y=30.0)
            
            rmse_result = tk.Label(
                text="0.007", bg="black",
                fg="white", font=("Arial-BoldMT", int(10.0)))
            rmse_result.place(x=680.0, y=70.0)
    
    elif(choose=='QNB') & (choose_algorithm == 'TSA'):
        product = 'QNBFB.IS'
        
        tsa_tomorrow(product)
        
        img_old=Image.open('QNB_tsa.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=tsa_result, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text="0.145", bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
        
    elif(choose=='KOC') & (choose_algorithm == 'TSA'):
        product = 'KCHOL.IS'
        
        tsa_tomorrow(product)
        
        img_old=Image.open('KOC_tsa.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=tsa_result, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text="0.079", bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
    
    elif(choose=='THY') & (choose_algorithm == 'TSA'):
        product = 'THYAO.IS'
        
        tsa_tomorrow(product)
        
        img_old=Image.open('THY_tsa.png')
        img_resized = img_old.resize((700,400))
        img = ImageTk.PhotoImage(img_resized)
        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=-35, y=-35)
        title.image = img
        
        predict_show = tk.Label(
            text="Tomorrow Prediction", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        predict_show.place(x=600.0, y=120.0)
        
        predict_result = tk.Label(
            text=tsa_result, bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        predict_result.place(x=645.0, y=160.0)
        
        rmse_show = tk.Label(
            text="RMSE", bg="black",
            fg="white", font=("Arial-BoldMT", int(15.0)))
        rmse_show.place(x=670.0, y=30.0)
        
        rmse_result = tk.Label(
            text="0.072", bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
        rmse_result.place(x=680.0, y=70.0)
    
    elif(choose=='BIM') & (choose_algorithm == 'TSA'):
       product = 'BIMAS.IS'
       
       tsa_tomorrow(product)
       
       img_old=Image.open('BIM_tsa.png')
       img_resized = img_old.resize((700,400))
       img = ImageTk.PhotoImage(img_resized)
       title = tk.Label(
           image=img, bg="black",
           fg="white", font=("Arial-BoldMT", int(20.0)))
       title.place(x=-35, y=-35)
       title.image = img
       
       predict_show = tk.Label(
           text="Tomorrow Prediction", bg="black",
           fg="white", font=("Arial-BoldMT", int(15.0)))
       predict_show.place(x=600.0, y=120.0)
       
       predict_result = tk.Label(
           text=tsa_result, bg="black",
           fg="white", font=("Arial-BoldMT", int(10.0)))
       predict_result.place(x=645.0, y=160.0)
       
       rmse_show = tk.Label(
           text="RMSE", bg="black",
           fg="white", font=("Arial-BoldMT", int(15.0)))
       rmse_show.place(x=670.0, y=30.0)
       
       rmse_result = tk.Label(
            text="0.173", bg="black",
            fg="white", font=("Arial-BoldMT", int(10.0)))
       rmse_result.place(x=680.0, y=70.0)
    
def option_selected_stock_product(variable):
    global choose
    choose=variable
    print(choose)
    
    return choose

def option_selected_algorithm(algorithm_select):
   global choose_algorithm
   choose_algorithm=algorithm_select
   print(choose_algorithm)
   
   return choose

def call_funcs():
    

    
    if(choose=='Select Your Product') | (choose_algorithm == 'Select An Algorithm'):
        error = tk.Label(window, 
        text = "Please Select An Option", bg='white', fg='black', font=("Arial-BoldMT", int(10.0))).place(x = 935,
        y = 480)
    elif(choose_algorithm == 'TSA'):
        img_old=Image.open('black.png')
        img_resized = img_old.resize((200,365))
        img = ImageTk.PhotoImage(img_resized)

        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=595, y=0)
        
        call_tsa()
    else:    
        img_old=Image.open('black.png')
        img_resized = img_old.resize((200,365))
        img = ImageTk.PhotoImage(img_resized)

        title = tk.Label(
            image=img, bg="black",
            fg="white", font=("Arial-BoldMT", int(20.0)))
        title.place(x=595, y=0)
        
        show_plot()
        call_product()
    









variable = tk.StringVar(window)
window.languages = ('Select Your Product','ASELSAN', 'KARDEMIR', 'QNB', 'KOC', 'THY', 'BIM' ) 
token_entry =ttk.OptionMenu(window, variable,*window.languages, command=option_selected_stock_product)
token_entry.place(x=820.0, y=140+25, width=321.0, height=35)
token_entry.focus()

algorithm_select = tk.StringVar(window)
window.algorithms = ('Select An Algorithm','LSTM', 'LSTM And NLP', 'TSA')
URL_entry = ttk.OptionMenu(window, algorithm_select,*window.algorithms, command=option_selected_algorithm)
URL_entry.place(x=820.0, y=220+25, width=321.0, height=35)





canvas.create_text(
    820.0, 156.0, text="Stock Products", fill="#515486",
    font=("Arial-BoldMT", int(13.0)), anchor="w")
canvas.create_text(
    820.0, 234.5, text="Algorithms", fill="#515486",
    font=("Arial-BoldMT", int(13.0)), anchor="w")
canvas.create_text(
    950.5, 88.0, text="Select The Attiributes",
    fill="#515486", font=("Arial-BoldMT", int(22.0)))

#%% SOL TARAF
title = tk.Label(
    text="Welcome to Yapay Finans", bg="#292b4d",
    fg="#e8e1ee", font=("Arial-BoldMT", int(25.0)))
title.place(x=27.0, y=30.0)

info_text = tk.Label(
    text="Yapay Finans is a forecasting application.\n"
    "As you see on the right section, \n"
    "there are some products and algorithms to choose.\n"
    "Application goal is predict the future of product\n"
    "which is choosen by you and with algorithm power\n"
    "Yapay Finans had created for test algorithms on forecasting\n"
    "We extremly recommend, do not trust results for investment\n\n\n"

    ,
    bg="#292b4d", fg="#e8e1ee", justify="left",
    font=("Georgia", int(16.0)))

info_text2 = tk.Label(
    text=
    "Created By\n\n",
    bg="#292b4d", fg="#fe5f55", justify="left",
    font=("Georgia", int(16.0)))

info_text3 = tk.Label(
    text=
    "Faruk Kirbac\t"
    "Salih Can Yavuz \t"
    "Bahadir Kelesoglu",
    bg="#292b4d", fg="#e8e1ee", justify="left",
    font=("Georgia", int(14.0)))

info_text.place(x=27.0, y=110.0)
info_text2.place(x=240.0, y=300.0)
info_text3.place(x=27.0, y=340.0)

img_old=Image.open('yapay_finans.png')
img_resized = img_old.resize((300,300))
img = ImageTk.PhotoImage(img_resized)

title = tk.Label(
    image=img, bg="#292b4d",
    fg="white", font=("Arial-BoldMT", int(20.0)))
title.place(x=225, y=400)


#%%

generate_btn_img = tk.PhotoImage(file=ASSETS_PATH / "generate.png")
generate_btn = tk.Button(
    image=generate_btn_img, borderwidth=0, highlightthickness=0,
    command=call_funcs, relief="flat")
generate_btn.place(x=915, y=401, width=180, height=55)

window.resizable(False, False)
window.mainloop()
