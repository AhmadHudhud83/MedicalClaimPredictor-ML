# Importing Dependencies
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
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
def target_encoder(df:pd.DataFrame,col:str,targetCol:str)->pd.DataFrame:

    try:

        #Calculating the mean target value (Amount) for each category in Specialty
        specialty_means = df.groupby(col)[targetCol].mean()

        # Mapping means to the Specialty column
        df[col] = df[targetCol].map(specialty_means)
        return df
    except Exception as e :
        raise Exception(f"Error in applying target encoder : {e}")

# Label Encoding Function for Gender & Marital Status
def label_encoder(df:pd.DataFrame,col:str)->pd.DataFrame:
    try:

        #For Gender Column
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])
    
        return df
    except Exception as e :
        raise Exception(f"Error in applying label encoder : {e}")


#One hot Encoding Function for Insurance Column
def one_hot_encoder(df:pd.DataFrame,col:str)->pd.DataFrame:
  try:
      
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(df[[col]])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]))
    df = pd.concat([df, encoded_df], axis=1)

    # Dropping the original Insurance column
    df.drop(col, axis=1, inplace=True)

    return df
  
  except Exception as e:
      raise Exception(f"Error in applying one hot encoder : {e}")

#Features Scaler Function




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
            "custom":lambda x :x**2.5,
            "sqrt":lambda x : x,
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
        

  


        
        #Applying transformation functions for training & testing sets

            
        train_df = transform(train_data,"Amount",transformation_type,apply_boxcox)
        test_df = transform(test_data,"Amount",transformation_type,apply_boxcox)

        # winsorization flag
        if apply_winsorization:
              train_df= winsorization(train_df,"Amount",lower_quantile,upper_quantile)
              test_df = winsorization(test_df,"Amount",lower_quantile,upper_quantile)
            

        #Applying encoding functions for training & testing sets

        # Label Encoding
        train_df = label_encoder(train_df,'Gender')
        test_df = label_encoder(test_df,'Gender')

        # One hot Encoding
        train_df = one_hot_encoder(train_df,"Insurance")
        test_df = one_hot_encoder(test_df,"Insurance")
        train_df = one_hot_encoder(train_df,"Specialty")
        test_df = one_hot_encoder(test_df,"Specialty")

    

        target_plots(train_df,"Amount","Training Dataset")

 
        #Creating folder for processed data
        
        os.makedirs(processed_data_path,exist_ok=True)

        # Storing Processed training & testing sets as outputs
     
        save_data(train_df,os.path.join(processed_data_path,"train_processed.csv"))
        save_data(test_df,os.path.join(processed_data_path,"test_processed.csv"))
        print(train_data.info())

        train_df.hist(bins =50 ,figsize=(12,8))
        plt.show()
    except Exception as e :
        raise Exception(f"An Error occured : {e}")


if __name__ == "__main__":
    main()


