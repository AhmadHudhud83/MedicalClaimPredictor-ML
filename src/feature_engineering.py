# Importing Dependencies
import pandas as pd
import numpy as np
import os

#Reading the Processed Train data
processed_train_data = pd.read_csv('./data/processed/train_processed.csv')
#Reading the Processed Test data 
processed_test_data = pd.read_csv('./data/processed/test_processed.csv')

# ...


# Storing Engineered Data 
data_path = os.path.join("data","engineered")
os.makedirs(data_path)

processed_train_data.to_csv(os.path.join(data_path,"train_engineered.csv"),index=False)
processed_test_data.to_csv(os.path.join(data_path,"test_engineered.csv"),index=False)


