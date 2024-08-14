# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import random
import functools
import logging
import os
import sys
import pickle
import time
import warnings
import pdb

from keras.models import load_model
import tempfile
import keras.models

# import FS_Final as fs

import os
import keras
import itertools
import matplotlib.pyplot as plt
from math import sqrt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from datetime import datetime
from itertools import product
from functools import reduce

#Importing SKlearn libraries for data preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, mean_squared_error,r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from matplotlib import pyplot
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

import keras.backend as K
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, subtract, dot, MaxPool2D, Flatten, AvgPool2D
from keras.layers import Input, concatenate, merge
from keras.callbacks import Callback
from keras import initializers
from keras import models
from keras.layers import Bidirectional
from keras.layers import Reshape

from keras.layers import Lambda
# from keras.backend import slice

pd.set_option('display.max_columns', None)





path = r'/content/drive/MyDrive/Nifty'

nifty = pd.read_csv('/content/drive/MyDrive/Nifty/Nifty50_indicators_final.csv')

nifty

shanghai = pd.read_csv(os.path.join(path,'000001.SS.csv'))

shanghai["shanghai"]=shanghai["Close"]
shanghai.drop(columns=["Open","High","Low","Adj Close","Volume","Close"],inplace=True)

shanghai.head()

oneyear = pd.read_csv("/content/drive/MyDrive/Nifty/1.csv")

fiveyear = pd.read_csv("/content/drive/MyDrive/Nifty/5.csv")

tenyear = pd.read_csv("/content/drive/MyDrive/Nifty/10.csv")

threemonth = pd.read_csv("/content/drive/MyDrive/Nifty/0.3.csv")

brazil=pd.read_csv("/content/drive/MyDrive/Nifty/brazil.csv")

brazil["brazil"]=brazil["Close"]
brazil.drop(columns=["Open","High","Low","Adj Close","Volume","Close"],inplace=True)

brazil

exrates=pd.read_csv("/content/drive/MyDrive/Nifty/exrates.csv")

exrates.head()

exrates['Date'] = pd.to_datetime(exrates['Date'], infer_datetime_format=True)





gold=pd.read_csv("/content/drive/MyDrive/Nifty/gold.csv")

gold.head()

gold["gold"]=gold["Price"]

gold.drop(columns=["Price","Open","High","Low","Vol.","Change %"],inplace=True)

nasdaq=pd.read_csv("/content/drive/MyDrive/Nifty/nasdaq.csv")

silver=pd.read_csv("/content/drive/MyDrive/Nifty/silver.csv")

nasdaq["nasdaq"]=nasdaq["Close"]
nasdaq.drop(columns=["Open","High","Low","Adj Close","Volume","Close"],inplace=True)

markets1=pd.merge(nasdaq,brazil, how="outer",on="Date")

markets=pd.merge(markets1,shanghai, how="outer", on="Date")

markets.isna().sum()

oneyear["1y"]=oneyear.Price

oneyear.drop(columns=["Open","High","Low","Change %","Price"],axis=1,inplace=True)

fiveyear["5y"]=fiveyear.Price

fiveyear.drop(columns=["Open","High","Low","Change %","Price"],axis=1,inplace=True)

tenyear["10y"]=tenyear.Price

tenyear.drop(columns=["Open","High","Low","Change %","Price"],axis=1,inplace=True)

threemonth["0.3m"]=threemonth.Price

threemonth.drop(columns=["Open","High","Low","Change %","Price"],axis=1,inplace=True)

bond1=pd.merge(oneyear,threemonth,on='Date',how='inner')

bond2=pd.merge(fiveyear,tenyear,on='Date',how="inner")

bond=pd.merge(bond1,bond2,on="Date",how="inner")

bond.head()

nifty.head(2)

nifty['Date'] = pd.to_datetime(nifty['Date'], format="%d-%m-%Y")
nifty.head(2)

bond['Date'] = pd.to_datetime(bond['Date'], infer_datetime_format=True)

nifty.dtypes

nifty=nifty.merge(bond,how="left",on="Date")

markets['Date'] = pd.to_datetime(markets['Date'], infer_datetime_format=True)

gold["Date"]=pd.to_datetime(gold["Date"],infer_datetime_format=True)

gold["gold"]=gold["gold"].str.replace(",","").astype(float)

nifty=nifty.merge(markets,how="left",on="Date")

nifty=nifty.merge(gold,how="left",on="Date")

nifty=nifty.merge(exrates,how='left',on="Date")

# nifty.isna().sum()

nifty.fillna(method='bfill', axis=0,inplace=True)

# nifty.isna().sum()

# nifty['INDEX'] = 'nifty'

# df_master = pd.concat([nifty,banknifty,it,pharma ,fin ,auto , metal , energy ,realty,fmcg ,infra ,media ,psubank, pse])
# df_master = pd.concat([nifty,banknifty,it])
df_master = nifty.copy()
# df_master.INDEX.unique()

# index = pd.unique(df_master.INDEX)
cols_list = list(nifty.columns)
date_list = list(pd.unique(nifty.Date))
# for i in

Y = nifty.copy()
Y.columns

Y = Y[['Date', 'Close']]
Y.head(2)

# df_master.dropna(axis = 0, inplace = True)
# df_master.isna().sum(), len(df_master.columns)
df_master.head(2)

index_split = int(0.7 *len(df_master))

df_master_train = df_master[:index_split]
df_master_test = df_master[index_split:]
# df_master[df_master['Date'] > '2017-11-27']
len(df_master_train)/ len(df_master), len(df_master_test)/ len(df_master)

df_master_train.to_excel('train_data.xlsx')

df_master_test.to_excel('test.xlsx')

len(df_master)

# df_master.iloc[int(0.7 *len(df_master))]

df_master.head(2)



X_data_array_train = []
Date_list_train = []
columns_to_drop = ['Date']
for i in df_master_train.groupby(by=['Date']):
    Date_list_train.append(i[0])
    temp_data_train = i[1].drop(columns = columns_to_drop)
    X_data_array_train.append(np.array(temp_data_train.values))
X_data_array_train = np.array(X_data_array_train)

X_data_array_train.shape, len(Date_list_train)

X_data_array_test = []
Date_list_test = []
columns_to_drop = ['Date']
for i in df_master_test.groupby(by=['Date']):
    Date_list_test.append(i[0])
    temp_data_test = i[1].drop(columns = columns_to_drop)
    X_data_array_test.append(np.array(temp_data_test.values))
X_data_array_test = np.array(X_data_array_test)

X_data_array_test.shape, len(Date_list_test)

1598 + 685

# X_data_array = X_data_array.reshape((X_data_array.shape[0]), X_data_array.shape[2])

X_data_array_train = X_data_array_train.reshape((X_data_array_train.shape[0]), X_data_array_train.shape[2])
X_data_array_test = X_data_array_test.reshape((X_data_array_test.shape[0]), X_data_array_test.shape[2])

# [i[0] for i in temp_data.values]

# X_data_array

X_final_train = []
Y_final_train = []
for i in range(len(Date_list_train)-20):
    temp_X_train = X_data_array_train[i:i+20,:]
#     print(temp_X.shape)
    X_final_train.append(temp_X_train)
    temp_y_train = X_data_array_train[i+20,4]
    Y_final_train.append(temp_y_train)
X_final_train = np.array(X_final_train)
Y_final_train = np.array(Y_final_train)

X_final_test = []
Y_final_test = []
for i in range(len(Date_list_test)-20):
    temp_X_test = X_data_array_test[i:i+20,:]
#     print(temp_X.shape)
    X_final_test.append(temp_X_test)
    temp_y_test = X_data_array_test[i+20,4]
    Y_final_test.append(temp_y_test)
X_final_test = np.array(X_final_test)
Y_final_test = np.array(Y_final_test)

# X_final = np.transpose(X_final, (0, 2, 1))

# Y_final_test

X_final_train.shape, Y_final_train.shape

X_final_test.shape, Y_final_test.shape

Y_final_train = Y_final_train.reshape((Y_final_train.shape[0]), 1, 1)
X_final_train = X_final_train.reshape((X_final_train.shape[0]), X_final_train.shape[1],X_final_train.shape[2], 1)

Y_final_train = Y_final_train.reshape((Y_final_train.shape[0]), 1, 1)
X_final_train = X_final_train.reshape((X_final_train.shape[0]), X_final_train.shape[1],X_final_train.shape[2], 1)


X_final_train.shape, Y_final_train.shape

Y_final_test = Y_final_test.reshape((Y_final_test.shape[0]), 1, 1)
X_final_test = X_final_test.reshape((X_final_test.shape[0]), X_final_test.shape[1],X_final_test.shape[2], 1)

Y_final_test = Y_final_test.reshape((Y_final_test.shape[0]), 1, 1)
X_final_test = X_final_test.reshape((X_final_test.shape[0]), X_final_test.shape[1],X_final_test.shape[2], 1)


X_final_test.shape, Y_final_test.shape

Y_final_train

# X_final_train

from tensorflow.keras import layers
from tensorflow.keras import regularizers

from keras.layers import LSTM, Embedding, ConvLSTM2D
from keras.layers import Bidirectional, GlobalMaxPool1D

X_final_train.shape

"""# Model Architecture"""

input_shape = (X_final_train.shape[1],X_final_train.shape[2], X_final_train.shape[3])
input_shape

from tensorflow.keras import layers
from tensorflow.keras import regularizers

# X_final = np.asarray(X_final).astype('float32')

# input layers for seq_1 and seq_2
input_seq_1 = Input(shape= input_shape)

conv_3_6 = Conv2D(1,kernel_size=(1,48), strides= 1, padding='valid', activation='relu', name ="3_6_convolution", kernel_initializer='glorot_uniform')
x1 = conv_3_6(input_seq_1)

x1.shape

reshape_1 = tf.keras.layers.Reshape((1, 20),input_shape=(20,1,1))

x1 = reshape_1(x1)

lstm_1_1 = Bidirectional(LSTM(units=256, return_sequences=True, activation='relu',
                              kernel_regularizer=regularizers.l2(1e-2), name ="1_1_LSTM"))
x1 = lstm_1_1(x1)

lstm_1_2 = Bidirectional(LSTM(units=64, return_sequences=True, activation='relu',
                              kernel_regularizer=regularizers.l2(1e-2), name ="1_2_LSTM"))
x1 = lstm_1_2(x1)

lstm_1_3 = Bidirectional(LSTM(units=32, return_sequences=True, activation='relu',
                              kernel_regularizer=regularizers.l2(1e-2), name ="1_3_LSTM"))
x1 = lstm_1_3(x1)



# lstm1_2 = ConvLSTM2D(64, return_sequences=True, kernel_size=(2,1))
# lstm2_2 = ConvLSTM2D(32, return_sequences=True, kernel_size=(1,1))
# # lstm3 = LSTM(16, return_sequences=True, input_shape=(97, 32))
# # lstm4 = LSTM(8, return_sequences=True, input_shape=(97, 16))
# # lstm5 = LSTM(4, return_sequences=True, input_shape=(80, 8))

# x1 = lstm1_2(x1)
# x1 = lstm2_2(x1)
# # x1_2_2 = Flatten(name = "x1_2_2")(x1_2_2)

# xdense1 = TimeDistributed(Dense(1024, name = "Dense_1_1", activation = "relu"))(x1)
xdense1 = TimeDistributed(Dense(512, name = "Dense_1_2", activation = "relu"))(x1)
xdense1 = Dropout(0.1)(xdense1)
xdense1 = TimeDistributed(Dense(256, name = "Dense_1_3", activation = "relu"))(xdense1)
xdense1= Dropout(0.1)(xdense1)
xdense1 = TimeDistributed(Dense(128, name = "Dense_1_4", activation = "relu"))(xdense1)
# xdense1= Dropout(0.1)(xdense1)
xdense1 = TimeDistributed(Dense(64, name = "Dense_1_5", activation = "relu"))(xdense1)
xdense1 = TimeDistributed(Dense(32, name = "Dense_1_6", activation= "relu"))(xdense1)
# xdense1 = Dropout(0.1)(xdense1)
xdense1 = TimeDistributed(Dense(16, name = "Dense_1_7", activation = "relu"))(xdense1)
# output_yfuture = TimeDistributed(Dense(1, name = "Dense_1_7", activation = "relu"))(xdense1)
output_yfuture = Dense(1, name = "Dense_final")(xdense1)

model = Model(inputs=input_seq_1, outputs = output_yfuture)

opti = keras.optimizers.Adam(lr = 0.005)
model.compile(optimizer = opti, loss = ["mae"], metrics = ["mae", "mse"])

model.summary()



"""

# Model Training"""

checkpointer = [EarlyStopping(monitor='val_loss', patience=80)]
batches = 512
epoch_num = 2000

history = model.fit(X_final_train, Y_final_train, batch_size=batches, epochs = epoch_num, validation_split= 0.3,callbacks= checkpointer)
# model.fit([train_paddedJourney, train_extra_data.values], [train_y, train_y], batch_size=batches, epochs = epoch_num, validation_split= 0.3,callbacks= checkpointer)
history

yhat_train = model.predict(X_final_train)
yhat_test = model.predict(X_final_test)

yhat_train

yhat_train.shape

test_pred=yhat_test.reshape(yhat_test.shape[0])

train_pred=yhat_train.reshape(yhat_train.shape[0])



# Y_final

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


yhat_train_array = np.array([yhat_train[i][0][0] for i in range(yhat_train.shape[0])])
yhat_test_array = np.array([yhat_test[i][0][0] for i in range(yhat_test.shape[0])])

Y_final_train_array = np.array([Y_final_train[i][0][0] for i in range(Y_final_train.shape[0])])
Y_final_test_array = np.array([Y_final_test[i][0][0] for i in range(Y_final_test.shape[0])])

# test_yhat_future  = np.array([yhat[i][0][0] for i in range(yhat.shape[0])])
len(yhat_train_array), len(yhat_test_array)

Y_final_train_array

# Y_final_test

"""# Model Results"""

plt.scatter(Y_final_train_array,yhat_train_array,alpha=0.5)
# plt.xlim(0,400)
# plt.ylim(0,400)
plt.show()
print('Test R-Square :',r2_score(Y_final_train_array, yhat_train_array))
mae_test = mean_absolute_error(Y_final_train_array,yhat_train_array)
rmse_test = sqrt(mean_squared_error(Y_final_train_array, yhat_train_array))
print('Test MAE      :',mae_test)
print('Test Mean     :',Y_final_train.mean())
print('Test MAPE     :',mae_test/Y_final_train.mean())
print('Test RMSE     :',rmse_test)

plt.scatter(Y_final_test_array,yhat_test_array,alpha=0.5)
# plt.xlim(0,400)
# plt.ylim(0,400)
plt.show()
print('Test R-Square :',r2_score(Y_final_test_array, yhat_test_array))
mae_test = mean_absolute_error(Y_final_test_array,yhat_test_array)
rmse_test = sqrt(mean_squared_error(Y_final_test_array, yhat_test_array))
print('Test MAE      :',mae_test)
print('Test Mean     :',Y_final_test.mean())
print('Test MAPE     :',mae_test/Y_final_test.mean())
print('Test RMSE     :',rmse_test)

from google.colab import drive
drive.mount('/content/drive')

Y['Close']

# Y_final_test, yhat_test_array

plt.figure(figsize=(15,10))
plt.plot(list(Y['Close']),alpha=1,color ='tab:blue')
# plt.plot(Y_final_test_array,alpha=1,color ='tab:blue')
plt.plot(yhat_train_array,alpha=1,color ='tab:green')
plt.plot(yhat_test_array,alpha=1,color ='tab:red')
plt.ylabel('Nifty')
plt.xlabel('Trading days')
plt.legend(["Real value","Predicted value"])


len(Y['Close'])

nifty_df_output_Train = pd.DataFrame(index = range(len(Y_final_train_array)), columns = ['Actual_Train', 'Predicted_Y_Train'])
nifty_df_output_Test = pd.DataFrame(index = range(len(Y_final_test_array)), columns = ['Actual_Test', 'Predicted_Y_Test'])
nifty_df_output_Train['Actual_Train'] = Y_final_train_array
nifty_df_output_Train['Predicted_Y_Train'] = yhat_train_array
nifty_df_output_Test['Actual_Test'] = Y_final_test_array
nifty_df_output_Test['Predicted_Y_Test'] = yhat_test_array

nifty_df_output_Test.head(2)

nifty_df_output_Test.to_csv('nifty_df_output_Test.csv', index = False)
nifty_df_output_Train.to_csv('nifty_df_output_Train.csv', index = False)

nifty_df_output['Actual'] = list(Y['Close'])
nifty_df_output['Predicted_Y_Train'] = list(Y['Close'])

list(Y['Close'])




index1=range(0,2261)

niftyexcel=pd.DataFrame(test_yhat_future,columns=["predicted"])

yhat=yhat.reshape(2263,-1)

niftyexcel1=pd.DataFrame(test_y,columns=["real"])

niftyexcel

niftyexcel1

sim=pd.concat([niftyexcel,niftyexcel1],axis=1)

sim.to_excel("sim.xlsx")

nifty.to_excel("excel.xlsx")

yhat

ss=pd.DataFrame(yhat)

ss.to_excel("ss.xlsx")

