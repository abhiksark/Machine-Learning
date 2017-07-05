# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import re

test_data = pd.read_csv("test_data.csv")
train_data = pd.read_csv("train_data.csv")

ID = pd.read_csv("ID.csv")

inspect_test = test_data.head()
inspect_train = train_data.head()
 
heading_test = list(test_data) #heading for data frame 
heading_train = list(train_data) 

test_data.drop(['Unnamed: 0'], inplace=True, axis=1)
train_data.drop(['Unnamed: 0'], inplace=True, axis=1)

y = np.array(train_data['segment'])



from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
data.head()