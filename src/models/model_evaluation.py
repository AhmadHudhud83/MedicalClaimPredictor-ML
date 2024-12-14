# Importing Dependencies

import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


# Loading Data Function
def load_data(filepath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error reading data from {filepath} : {e}")
    
# Preparing Training & Testing Datasets function
def prepare_data(df:pd.DataFrame)->tuple[pd.DataFrame,pd.Series]:
    try:
        x_train = df.drop(columns=["Amount"],axis= 1)
        y_train = df["Amount"]
        return x_train , y_train
    except Exception as e:
        raise Exception(f"Error Preparing Datasets : {e}")


# Loading the regression model  
def load_model(filepath:str)->RandomForestRegressor:
    try:
        with open(filepath,"rb") as f:
            model=  pickle.load(f)
        return model
    except Exception as e:
        raise Exception(print(f"Error loading the model from {filepath} : {e}"))
    
# Model Evaluation function
def evaluation_model(model:RandomForestRegressor,X_test: pd.DataFrame,Y_test :pd.Series)->dict:
    try:
        # Predicting test set
        test_data_prediction = model.predict(X_test)

        # Defining r square measure
        r2_test = r2_score(Y_test, test_data_prediction)

        # For debugging
        print(X_test.shape)
        print('R squared value (Test): ', r2_test)


        # Plotting predicted vs actual values

        plt.figure(figsize=(10, 6))
        plt.scatter(Y_test, test_data_prediction, color='green', alpha=0.5)
        plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linewidth=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values (Test Set)')
        plt.show()

        
        # Returning metrics dictionary
        metrics_dict= {
                 "r2_test":r2_test
            }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error occured during model evaluation : {e}")


# Saving Metrics functions
def save_metrics(metrics_dict:dict,metrics_path:str)->None:
    try:
        with open(metrics_path,'w') as f:
         json.dump(metrics_dict,f,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path} : {e}")

# Main function
def main():
    try:
        # Defining data , model and metrics paths
        test_data_path = "./data/interim/test_engineered.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"

        # Loading teset data
        test_data =load_data(test_data_path)

        # Defining X-test & Y-test
        X_test,Y_test = prepare_data(test_data)

        # Loading Model
        model = load_model(model_path)

        # Getting metrics based on model and test set
        metrics = evaluation_model(model,X_test, Y_test)

        # Saving results of metrics
        save_metrics(metrics,metrics_path)
    except Exception as e:
        raise Exception(f"Error Occured  : {e}")



if __name__ == "__main__":
    main()