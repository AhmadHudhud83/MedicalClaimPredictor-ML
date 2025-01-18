# Importing Dependencies
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


train_data = pd.read_csv('./data/raw/train.csv')
# Reading the Test data 
test_data = pd.read_csv('./data/raw/test.csv')


def handle_missing_values(df):
    # Replace "Unknown" in Marital Status with the most frequent value
    most_frequent_category = df["Marital Status"].mode()[0]
    df["Marital Status"] = df["Marital Status"].replace("Unknown", most_frequent_category)
    
    # "Unknown" in Insurance is treated as a separate category
    df["Insurance"] = df["Insurance"].replace("Unknown", "Unknown Category")

    return df

# Function for Target Encoding of Specialty
def target_encoder(df):
    specialty_means = df.groupby("Specialty")['Amount'].mean()
    df['Specialty'] = df["Specialty"].map(specialty_means)
    return df

# Function for Label Encoding Gender & Marital Status
def label_encoder(df):
    # Encoding 'Gender' as 1 for 'Male' and 0 for 'Female'
    df.replace({'Gender': {'Male': 1, 'Female': 0}}, inplace=True)
    
    # Encoding 'Marital Status' to numerical categories
    df.replace({'Marital Status': {'Married': 4, 'Divorced': 3, 'Single': 2, 'Widowed': 1}}, inplace=True)
    return df

# Function for One Hot Encoding of Insurance
def one_hot_encoder(df):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(df[['Insurance']])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Insurance']))
    df = pd.concat([df, encoded_df], axis=1)
    df.drop('Insurance', axis=1, inplace=True)
    return df

# Function for Scaling Numeric Features
def scaler(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

# Applying missing values handler to training & testing data
train_data_processed = handle_missing_values(train_data)
test_data_processed = handle_missing_values(test_data)

# Applying encoding functions to training & testing data
encoding_funcs = [target_encoder, label_encoder, one_hot_encoder]

for func in encoding_funcs:
    train_data_processed = func(train_data_processed)
    test_data_processed = func(test_data_processed)

# Scaling numeric features
features_to_scale = ["Severity", "Age", "Marital Status", "Specialty"]
scaler(train_data_processed, features_to_scale)
scaler(test_data_processed, features_to_scale)

# Checking if missing values in Marital Status were handled correctly
unique_values = train_data_processed['Marital Status'].unique()
print("Unique values in Marital Status after processing:", unique_values)

# Displaying some samples of the preprocessed data
print(train_data_processed.head())

# Creating folder for processed data
data_path = os.path.join("data", "processed")
os.makedirs(data_path, exist_ok=True)

# Storing processed training & testing sets
train_data_processed.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
test_data_processed.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
