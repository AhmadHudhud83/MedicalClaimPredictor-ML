# Importing Dependencies
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor


#Reading the Processed Train data
train_data = pd.read_csv('./data/engineered/train_engineered.csv')
#Reading the Processed Test data 
test_data = pd.read_csv('./data/engineered/test_engineered.csv')

#
x_train = train_data.iloc[:, 1:].values 
y_train = train_data.iloc[:, 0].values
print(x_train.shape)
print(y_train.shape)

#Applying RandomForest Regressor to fit the data
rfg = RandomForestRegressor(bootstrap=True,n_estimators=271,max_depth=13,min_samples_leaf=1,min_samples_split=18)

rfg.fit(x_train, y_train)

pickle.dump(rfg,open("model.pkl","wb"))
