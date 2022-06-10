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
