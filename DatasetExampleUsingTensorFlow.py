#!/usr/bin/env python
# coding: utf-8

# In[11]:

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Importing Data File from Local folder
dataframe = pd.read_csv(r"C:/Users/Sumanth/Desktop/basicdata.csv")
#Displays the top 5 rows from the file
print(dataframe.head())
'''
Splits the Dataframe into Training and Test data. test_size is the percentage of data which splits into Test data, 
and the remaining gets assigned to Training data
'''
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# Method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=False):
  #Converting the Datafram to dictionary because it is not possible to convert a non-rectangular Python sequence to Tensor
  ds = tf.data.Dataset.from_tensor_slices(dict((dataframe)))
  return ds

train_ds= df_to_dataset(test)
print (train_ds)

train_ds= df_to_dataset(train)
print (train_ds)

train_ds= df_to_dataset(val)
print (train_ds)
