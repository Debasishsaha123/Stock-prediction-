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


# In[8]:


training_set=df['Open']
training_set=pd.DataFrame(training_set)


# In[12]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)


# In[16]:


training_set


# In[15]:


training_set_scaled


# In[ ]:




