# Importing Dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
# Loading parameters function from params.yaml file
def load_params(params_path:str):
    try:

        with open(params_path,"r") as f:
            params = yaml.safe_load(f)
        return params
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
        X = df.drop('Amount',axis= 1)
        Y = df["Amount"]
        return X , Y
    except Exception as e:
        raise Exception(f"Error Preparing Datasets : {e}")

# Get Model function
def get_model(model_name:str,**kwargs):
    models = {
        "random_forest":RandomForestRegressor,
        "decision_tree":DecisionTreeRegressor ,
        "xgboost":xgb.XGBRegressor
    }
    
    if model_name not in models:
        raise ValueError(f"Unsupported model : {model_name}")
    return models[model_name](**kwargs)

        
        
# Training Model function 
def train_model(x_train:pd.DataFrame, y_train : pd.Series,model_name:str,kwargs:dict):
    
    try:
        #Applying the model
        model = get_model(model_name,**kwargs)

        # Performing cross validation
        scores = cross_val_score(model,x_train,y_train,cv=5,scoring='r2')
        model.fit(x_train, y_train)
        print(f"Cross-validation scores: {scores}") 
        print(f"Mean cross-validation score: {np.mean(scores)}")
        return model
    except Exception as e:
        print(f"Error in training the model : {e}")

# Saving Model function
def save_model(model, filepath:str) -> None:
    try:
        with open(filepath,"wb") as f:

            joblib.dump(model,f)

    except Exception as e:
        raise Exception(f"Error occured during saving the model to path : {filepath}  : {e}")


def main():
    try:
        # Defining paths & model name
        params_path = "params.yaml"
        train_data_path = "./data/interim/train_engineered.csv"
        test_data_path = "./data/interim/test_engineered.csv"
        model_save_path = "models/model.joblib"

        # Loading hyperparameters    
        params = load_params(params_path)
        model_name = params["model_building"]["model_name"]
        model_kwargs = params["model_building"]["kwargs"]
     
        # Loading the train dataset 
        train_dataset =load_data(train_data_path)
        test_dataset = load_data(test_data_path)
      
        # Preparing x_train , y_train data for fitting
        x_train , y_train = prepare_data(train_dataset)
        x_test,y_test = prepare_data(test_dataset)

    

    

      # Debugging
        print(f"Train features shape: {x_train.shape}")
        print(f"Train target shape: {y_train.shape}")
        print(f"Test features shape: {x_test.shape}")
        print(f"Test target shape: {y_test.shape}")

        # Training the model & fitting the data
        model = train_model(x_train=x_train,y_train=y_train,kwargs=model_kwargs,model_name=model_name)

        # Saving the model
        save_model(model,model_save_path)

    except Exception as e:
        raise Exception(f"An Error Occured : {e}")
    
  

if __name__ == "__main__":
    main()


