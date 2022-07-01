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


# In[11]:


#creating a data structure with 100 time steps and 1 output
X_train=[]
y_train=[]
for i in range(100,1258):
    X_train.append(training_set_scaled[i-100:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)


# In[12]:


#reshaping 
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


# In[13]:


X_train.shape


# In[14]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


# In[15]:


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


# In[16]:


model.compile(optimizer='adam',loss=tf.keras.losses.mse)


# In[17]:


model.fit(X_train,y_train,epochs=100,batch_size=32)


# In[20]:


df_test=pd.read_csv('Google_Stock_Price_Test.csv',index_col='Date',parse_dates=True)


# In[21]:


df_test.head()


# In[36]:


df_test.shape


# In[22]:


df_test.info()


# In[24]:


df_test['Volume']=df_test['Volume'].str.replace(',','').astype(float)


# In[25]:


df.info()


# In[44]:


real_stock_price=df_test.iloc[:,1:2].values


# In[26]:


test_set=df_test['Open']
test_set=pd.DataFrame(test_set)


# In[28]:


test_set.head()


# In[30]:


#getting the predicted stock price of 2017
dataset_total=pd.concat((df['Open'],df_test['Open']),axis=0)


# In[31]:


dataset_total


# In[32]:


dataset_total.shape


# In[35]:


dataset_total.tail(12)


# In[39]:


inputs=dataset_total[len(dataset_total)-len(df_test)-100:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
X_test=[]
for i in range(100,120):
    X_test.append(inputs[i-100:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price=model.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)


# In[40]:


predicted_stock_price=pd.DataFrame(predicted_stock_price)


# In[41]:


predicted_stock_price


# In[42]:


predicted_stock_price.info()


# In[46]:


#visualize the result
plt.plot(real_stock_price,color='r',label='Real google price stock')
plt.plot(predicted_stock_price,color='g',label='Predicted google price stock')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()


# In[ ]:




