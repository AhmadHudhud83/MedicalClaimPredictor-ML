# Importing Dependencies
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

#Loading the data from csv file to a Pandas DataFrame
dataset = pd.read_csv(r"C:\Users\Gigabyte\Desktop\medicalmalpractice.csv")

#Printing first 5 samples of data
print(dataset.head())

#Splitting Data into Training Set & Testing Set
train_data,test_data = train_test_split(dataset,test_size=0.3,random_state=101)

#Create data folder 
data_path = os.path.join("data","raw")
os.makedirs(data_path)

#Storing training & testing sets as csv files
train_data.to_csv(os.path.join(data_path,"train.csv"),index=False)
test_data.to_csv(os.path.join(data_path,"test.csv"),index=False)
