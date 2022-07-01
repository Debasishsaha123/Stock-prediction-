#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np


# In[2]:


df=pd.read_csv('Google_Stock_Price_Train.csv',index_col='Date',parse_dates=True)


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isna().any()


# In[6]:


#Changing the datatype of close and volume t ofloat
df['Close']=df['Close'].str.replace(',','').astype(float)
df['Volume']=df['Volume'].str.replace(',','').astype(float)


# In[7]:


training_set=df['Open']
training_set=pd.DataFrame(training_set)


# In[8]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)


# In[9]:


training_set


# In[10]:


training_set_scaled


# In[12]:


#creating a data structure with 100 time steps and 1 output
X_train=[]
y_train=[]
for i in range(100,1258):
    X_train.append(training_set_scaled[i-100:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)


# In[14]:


#reshaping 
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


# In[15]:


X_train.shape


# In[16]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


# In[18]:


#creating the model RNN
model = Sequential()
#adding the first LSTM layer
model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))

#adding the second LSTM layer
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

#adding the third LSTM layer
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

#adding the fourth LSTM layer
model.add(LSTM(units=50))
model.add(Dropout(0.2))

#adding the output layer
model.add(Dense(units=1))


# In[19]:


model.compile(optimizer='adam',loss=tf.keras.losses.mse)


# In[20]:


model.fit(X_train,y_train,epochs=100,batch_size=32)


# In[ ]:




