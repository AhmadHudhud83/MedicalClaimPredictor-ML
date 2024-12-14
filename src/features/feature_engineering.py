# Importing Dependencies
import pandas as pd
import numpy as np
import os



#Reading the Processed Train & Test Datasets
def load_data(filepath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} : {e}")



# Saving Dataset function as an output of the feature engineering stage
def save_data(df:pd.DataFrame,filepath:str)->None:
    try:

     df.to_csv(filepath,index=False)
    
    except Exception as e:
        raise Exception(f"Error saving data to {filepath} : {e}")






def main():
    # General Exception for main function

    try:
        # Defining data path
        processed_data_path = "./data/processed"
        engineered_data_path = "./data/interim"

        #Loading the Processed Datasets
        processed_train_data=  load_data(os.path.join(processed_data_path,"train_processed.csv"))
        processed_test_data =  load_data(os.path.join(processed_data_path,"test_processed.csv"))

        # Engineered Data Code #

        
        engineered_train_data = processed_train_data
        engineered_test_data = processed_test_data


        # Engineered Data Code #

        #Creating folder for Engineered Data
        os.makedirs(engineered_data_path)

        # Storing Engineered Data
        save_data(engineered_train_data,os.path.join(engineered_data_path,"train_engineered.csv"))
        save_data(engineered_test_data,os.path.join(engineered_data_path,"test_engineered.csv"))


    except Exception as e:
        raise Exception(f"An Error Occured : {e}")
    

if __name__ == "__main__":
    main()

