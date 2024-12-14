# Importing Dependencies
import pandas as pd
import numpy as np
import os
import pickle
import yaml
from sklearn.ensemble import RandomForestRegressor

# Loading parameters function from params.yaml file
def load_params(params_path:str)->int:
    try:

        with open(params_path,"r") as f:
            params = yaml.safe_load(f)
        return params["model_building"]["n_estimators"]
    except Exception as e :
        raise Exception(f"Error loading parameters from {params_path}  : {e}")


# Loading Engineered Datasets function
def load_data(filepath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} : {e}")

# Preparing Training & Testing Datasets function
def prepare_data(df:pd.DataFrame)->tuple[pd.DataFrame,pd.Series]:
    try:
        x_train = df.drop(columns=["Amount"],axis= 1)
        y_train = df["Amount"]
        return x_train , y_train
    except Exception as e:
        raise Exception(f"Error Preparing Datasets : {e}")


# Training Model function 
def train_model(x_train:pd.DataFrame, y_train : pd.Series,n_estimators:int)->RandomForestRegressor:
    
    try:
        #Applying RandomForest Regressor to fit the data

        rfg = RandomForestRegressor(bootstrap=True,n_estimators=n_estimators,max_depth=13,min_samples_leaf=1,min_samples_split=18)
        rfg.fit(x_train, y_train)
        return rfg
    except Exception as e:
        print(f"Error in training the model : {e}")

# Saving Model function
def save_model(model:RandomForestRegressor, filepath:str) -> None:
    try:
        with open(filepath,"wb") as f:

            pickle.dump(model,f)

    except Exception as e:
        raise Exception(f"Error occured during saving the model to path : {filepath}  : {e}")
def main():
    try:
        # Defining paths & model name
        params_path = "params.yaml"
        train_data_path = "./data/interim/train_engineered.csv"
        test_data_path = "./data/interim/test_engineered.csv"
        model_save_path = "models/model.pkl"

        # Loading hyperparameters 
        n_estimators =load_params(params_path)

        # Loading the train dataset
        train_dataset =load_data(train_data_path)
       
        # Preparing x_train , y_train data for fitting
        x_train , y_train = prepare_data(train_dataset)
        
        # Training the model & fitting the data
        model = train_model(x_train,y_train,n_estimators)
        
        # Saving the model
        save_model(model,model_save_path)

    except Exception as e:
        print(f"An Error Occured : {e}")
    
    # Printing shape for debugging issues
        print(x_train.shape)
        print(y_train.shape)

    

if __name__ == "__main__":
    main()


