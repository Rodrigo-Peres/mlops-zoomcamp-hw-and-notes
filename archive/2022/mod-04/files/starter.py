#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip3 install pandas pyarrow fastparquet


# In[2]:


#get_ipython().system('pip freeze | grep scikit-learn')


# In[3]:


import sys
import pickle
import numpy as np
import pandas as pd


# In[4]:

year = sys.argv[1]
month = sys.argv[2]


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[5]:


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[6]:


df = read_data(f"fhv_tripdata_{year}-{month}.parquet")


# In[7]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# In[8]:


y_pred


# In[9]:


print(f"Mean ride duration predictions {np.mean(y_pred)}")


# In[ ]:


#year = 2021
#month = 2

#df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[ ]:


#df['predictions'] = y_pred


# In[ ]:


#df_results = df[['ride_id', 'predictions']]


# In[ ]:


#output_file = "hw04_data/ride_duration_predictions.parquet"

#df_results.to_parquet(
#    output_file,
#    engine='pyarrow',
#    compression=None,
#    index=False
#)

