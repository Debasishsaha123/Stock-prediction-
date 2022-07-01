#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np


# In[7]:


df=pd.read_csv('Google_Stock_Price_Train.csv',index_col='Date',parse_dates=True)


# In[8]:


df.head()


# In[9]:


df.shape


# In[11]:


df.isna().any()


# In[12]:


df.info()


# In[13]:


df['Open'].plot(figsize=(16,6))


# In[14]:


df['High'].plot(figsize=(16,6))


# In[17]:


#Changing the datatype of close and volume t ofloat
df['Close']=df['Close'].str.replace(',','').astype(float)
df['Volume']=df['Volume'].str.replace(',','').astype(float)


# In[18]:


df.info()


# In[19]:


df['Close'].plot(figsize=(16,6))


# In[21]:


#7 day rolling mean
df.rolling(7).mean().head(20)
#Pandas dataframe.rolling() function provides the feature of rolling window calculations. 
#The concept of rolling window calculation is most primarily used in signal processing and time-series data.


# In[22]:


df['Open'].plot(figsize=(16,6),color='r')
df.rolling(window=30).mean()['Close'].plot(color='g')


# In[23]:


df['Close: 30 day mean']=df.rolling(window=30).mean()['Close']
df[['Close','Close: 30 day mean']].plot(figsize=(16,6))


# In[24]:


#optimal specify a mnimum number of periods
df['Close'].expanding(min_periods=1).mean().plot(figsize=(16,6),color='g')


# In[25]:


training_set=df['Open']
training_set=pd.DataFrame(training_set)


# In[ ]:




