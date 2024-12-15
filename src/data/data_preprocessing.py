# Importing Dependencies
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from scipy.stats import boxcox

#Reading the Train & Test Datasets
def load_data(filepath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} : {e}")


#Based on performed EDA previously , there is an unkown/missing values for Marital Status column and Insurance Column 
#"Unknown" in Insurance column will be treated  as a Separate Category , since it represents 30% of values and it have a very high impact on target label
#"Unkown" in Matrial Status will be replaced with most frequent value , since it does not represent a high percantage of values , and it got less impact on target label


#Function for handling missing values of Marital Status
def handle_missing_values(df:pd.DataFrame) ->pd.DataFrame:
    
    try:

        # Finding most frequent value to replace with
        most_frequent_category= df["Marital Status"].mode()[0]
        
        # Replace missing values/Unkowns with most_frequent_category
        df["Marital Status"] = df["Marital Status"].replace("Unknown", most_frequent_category) 

        return df

    except Exception as e :
        raise Exception(f"Error filling missing values : {e}")

    


# Saving Dataset function as an output of the stage
def save_data(df:pd.DataFrame,filepath:str)->None:
    try:

     df.to_csv(filepath,index=False)
    
    except Exception as e:
        raise Exception(f"Error saving data to {filepath} : {e}")



#Features Encoding

# Target Encoding for Specialty Column , since it gives slightly better results on any model performance,
# and the splitted train set off test set will prevent data leakge.

#Function for Target Encoding
def target_encoder(df:pd.DataFrame)->pd.DataFrame:

    try:

        #Calculating the mean target value (Amount) for each category in Specialty
        specialty_means = df.groupby("Specialty")['Amount'].mean()

        # Mapping means to the Specialty column
        df['Specialty'] = df["Specialty"].map(specialty_means)
        return df
    except Exception as e :
        raise Exception(f"Error in applying target encoder : {e}")

# Label Encoding Function for Gender & Marital Status
def label_encoder(df:pd.DataFrame)->pd.DataFrame:
    try:

        #For Gender Column
        df.replace({'Gender':{'Male':1,'Female':0}},inplace=True)
        
        #For Marital Status Column
        df.replace({'Marital Status':{'Married':4,'Divorced':3,'Single':2,'Widowed':1}},inplace=True )
        return df
    except Exception as e :
        raise Exception(f"Error in applying label encoder : {e}")


#One hot Encoding Function for Insurance Column
def one_hot_encoder(df:pd.DataFrame)->pd.DataFrame:
  try:
      
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(df[['Insurance']])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Insurance']))
    df = pd.concat([df, encoded_df], axis=1)

    # Dropping the original Insurance column
    df.drop('Insurance', axis=1, inplace=True)




        #Assuming 'dataset' is your original dataframe

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid the dummy variable trap

    # Reshape the 'Specialty' column to a 2D array (necessary for sklearn)
    specialty_encoded = encoder.fit_transform(df[['Specialty']])

    # Convert the encoded array into a DataFrame with column names as the unique values in 'Specialty'
    encoded_df = pd.DataFrame(specialty_encoded, columns=encoder.get_feature_names_out(['Specialty']))

    # Concatenate the encoded columns with the original dataset (excluding the original 'Specialty' column)
    df = pd.concat([df.drop('Specialty', axis=1), encoded_df], axis=1)

    # Checking the first few rows of the updated dataset
    # df_encoded.head()
    return df
  
  except Exception as e:
      raise Exception(f"Error in applying one hot encoder : {e}")

#Features Scaler Function
def scaler(df:pd.DataFrame,features:list)->None:
    try:

     scaler = StandardScaler()
     df[features] = scaler.fit_transform(df[features])
     return df
    except Exception as e:
        raise Exception(f"Error in scaling features : {features}  : {e}")
    


# Winsorization function for the target variable
def winsorization(df: pd.DataFrame ,col:str , lower_quantile=0.17 , upper_quantile=0.88) ->pd.DataFrame:
  try:
        lower_bound = df[col].quantile(lower_quantile)
        upper_bound = df[col].quantile(upper_quantile)
        df[col] = df[col].clip(lower_bound,upper_bound)
        return df
  
  except Exception as e:
      raise Exception(f"Error in winsorization on column : {col} , with values of lower_quantile = {lower_quantile} , upper_quantile = {upper_quantile} :   {e}")


#Applying square transforming function to whole target variable

def transform(df : pd.DataFrame,col:str,transformation_type,apply_boxcox:bool) ->pd.DataFrame:

     
    transformation_types = {
            "square":lambda x: np.square(x),
            "cube":lambda x :x**2.5,
            "sqrt":lambda x : np.sqrt(x),
            "log2":lambda x :np.log2(x),
            
     }

    # boxcox transformation flag
    if apply_boxcox:
        try:
            df[col],fittedlambada = boxcox(df[col])
        except Exception as e:
            raise ValueError(f"Error applying Box-Cox transformation:  {e}")
    try:
        df[col] = transformation_types[transformation_type](df[col])
    except Exception as e:
        raise ValueError(f"Error applying {transformation_type} transformation: {e}")

    if transformation_type not in transformation_types:
            raise ValueError(f"Unsupported transformation type : {transformation_type}")

    return df

    
       
   
       
# target plotting for box plot & histogram  to evaluate  outliers & distribution of the target variable
def target_plots(df:pd.DataFrame,col:str,label:str)->None:
    
    try:
    # Showing boxplot to detect outliers
        plt.figure(figsize=(8, 4))
        sns.boxplot(df[col])
        plt.title(label)
        plt.ylabel(col)
        plt.show()

    #Showing the histogram of distribution

        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, bins=30)  
        plt.title("Amount Distribution")
        plt.xlabel("Amount")
        plt.ylabel("Frequency")
        plt.show()
    except Exception as e:
        raise Exception(f"Error plotting box plot for column : {col} :  {e}")

# Loading hyperparameters
def load_params(params_path:str):
    try:

        with open(params_path,"r") as f:
            params = yaml.safe_load(f)
        return params
    except Exception as e :
        raise Exception(f"Error loading parameters from {params_path}  : {e}")



def main():
    # General Excepetion
    try:

        # Defining data path
        raw_data_path = "./data/raw"
        processed_data_path = "./data/processed"

        #Loading params
        params_path = "params.yaml"
        params = load_params(params_path)
        apply_winsorization =params["data_preprocessing"]["apply_winsorization"]
        upper_quantile = params["data_preprocessing"]["upper_quantile"]
        lower_quantile = params["data_preprocessing"]["lower_quantile"]
        transformation_type = params["data_preprocessing"]["transformation_type"]
        apply_boxcox = params["data_preprocessing"]["apply_boxcox"]

        # Loading Datasets
        train_data = load_data(os.path.join(raw_data_path,"train.csv"))
        test_data = load_data(os.path.join(raw_data_path,"test.csv"))
        

        # Applying missing values handlers for training & testing sets
        train_data_processed = handle_missing_values(train_data)
        test_data_processed = handle_missing_values(test_data)


        
        #Applying transformation functions for training & testing sets

            
        train_data_processed= transform(train_data_processed,"Amount",transformation_type,apply_boxcox)
        test_data_processed = transform(test_data_processed,"Amount",transformation_type,apply_boxcox)

        # winsorization flag
        if apply_winsorization:
              train_data_processed= winsorization(train_data_processed,"Amount",lower_quantile,upper_quantile)
              test_data_processed = winsorization(test_data_processed,"Amount",lower_quantile,upper_quantile)
            

        #Applying encoding functions for training & testing sets
        encoding_funcs = [label_encoder,one_hot_encoder]

        for function in encoding_funcs:
            train_data_processed= function(train_data_processed)
            test_data_processed=function(test_data_processed)

        #Applying scaler function for training & testing sets

        features_to_scale = ["Severity","Age","Marital Status"]
        train_data_processed= scaler(train_data_processed,features_to_scale)
        test_data_processed=  scaler  (test_data_processed,features_to_scale)



        #Checking if 4 value is removed from Martial Status column
        unique_values = train_data_processed['Marital Status'].unique()
        print(unique_values)

        #Checking some samples of preprocessed data
        print(train_data_processed.head())

        #Checking Outliers for target variable using box plot
    

        target_plots(train_data_processed,"Amount","Training Dataset")



        #Creating folder for processed data
        
        os.makedirs(processed_data_path,exist_ok=True)

        # Storing Processed training & testing sets as outputs
     
        save_data(train_data_processed,os.path.join(processed_data_path,"train_processed.csv"))
        save_data(test_data_processed,os.path.join(processed_data_path,"test_processed.csv"))
        print(train_data_processed.info())

        train_data_processed.hist(bins =50 ,figsize=(12,8))
        plt.show()
    except Exception as e :
        raise Exception(f"An Error occured : {e}")


if __name__ == "__main__":
    main()


