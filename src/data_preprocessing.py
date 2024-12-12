# Importing Dependencies
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
#Reading the Train data
train_data = pd.read_csv('./data/raw/train.csv')
#Reading the Test data 
test_data = pd.read_csv('./data/raw/test.csv')

#Based on performed EDA previously , there is an unkown/missing values for Marital Status column and Insurance Column 
#"Unknown" in Insurance column will be treated  as a Separate Category , since it represents 30% of values and it have a very high impact on target label
#"Unkown" in Matrial Status will be replaced with most frequent value , since it does not represent a high percantage of values , and it got less impact on target label


#Function for handling missing values of Marital Status
def handle_missing_values(df):
    
    # Finding most frequent value to replace with
    most_frequent_category= df["Marital Status"].mode()[0]
    
    # Replace missing values/Unkowns with most_frequent_category
    df["Marital Status"] = df["Marital Status"].replace("Unknown", most_frequent_category) # 4 stands for Unkown Marital Status 
    
    

    return df

#Features Encoding

# Target Encoding for Specialty Column , since it gives slightly better results on any model performance,
# and the splitted train set off test set will prevent data leakge.

#Function for Target Encoding
def target_encoder(df):

    #Calculating the mean target value (Amount) for each category in Specialty
    specialty_means = df.groupby("Specialty")['Amount'].mean()

    # Mapping means to the Specialty column
    df['Specialty'] = df["Specialty"].map(specialty_means)
    return df
    

# Label Encoding Function for Gender & Marital Status
def label_encoder(df):
    #For Gender Column
    df.replace({'Gender':{'Male':1,'Female':0}},inplace=True)
    
    #For Marital Status Column
    df.replace({'Marital Status':{'Married':4,'Divorced':3,'Single':2,'Widowed':1}},inplace=True )
    return df


#One hot Encoding Function for Insurance Column
def one_hot_encoder(df):
  encoder = OneHotEncoder(sparse_output=False)
  encoded_data = encoder.fit_transform(df[['Insurance']])
  encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Insurance']))
  df = pd.concat([df, encoded_df], axis=1)

  # Dropping the original Insurance column
  df.drop('Insurance', axis=1, inplace=True)
  return df

#Features Scaler Function
def scaler(df,features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    


# Winsorization function for the target variable
def winsorization(df,col, lower_quantile=0.17, upper_quantile=0.88):
    lower_bound = df[col].quantile(lower_quantile)
    upper_bound = df[col].quantile(upper_quantile)
    df[col] = df[col].clip(lower_bound,upper_bound)
    return df

#Applying square transforming function to whole target variable
def transform(df,col,type="square"):
    
    if type =="square":
        df[col] = np.square(df[col] )
    elif type =="sqrt":
        df[col] = np.sqrt(df[col] )
  
    
    return df

# Box plot function to detect outliers of the target
def box_plot(df,col,label):
  
    plt.figure(figsize=(8, 4))
    sns.boxplot(df[col])
    plt.title(label)
    plt.ylabel(col)
    plt.show()


#Applying missing values handlers for training & testing sets
train_data_processed=handle_missing_values(train_data)
test_data_processed=handle_missing_values(test_data)


#Applying transformation functions for training & testing sets

transform_funcs = [winsorization,transform]

for f in transform_funcs:
    train_data_processed= f(train_data_processed,"Amount")
    test_data_processed = f(test_data_processed,"Amount")

#Applying encoding functions for training & testing sets
funcs = [target_encoder,label_encoder,one_hot_encoder]

for f in funcs:
    train_data_processed= f(train_data_processed)
    test_data_processed=f(test_data_processed)

#Applying scaler function
features_to_scale = ["Severity","Age","Marital Status","Specialty"]
scaler(train_data_processed,features_to_scale)
scaler(test_data_processed,features_to_scale)



#Checking if 4 value is removed from Martial Status column
unique_values = train_data_processed['Marital Status'].unique()
print(unique_values)

#Checking some samples of preprocessed data
print(train_data_processed.head())

#Checking Outliers for target variable using box plot


box_plot(train_data_processed,"Amount","Training Dataset")
box_plot(test_data_processed,"Amount","Testing Dataset")


#Creating folder for processed data
data_path = os.path.join("data","processed")
os.makedirs(data_path)

# Storing Processed training & testing sets as outputs
print("HERE BODY ! , THE SHAPE AFTER PREPROCESSING !! ",train_data_processed.shape)
train_data_processed.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
test_data_processed.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)
