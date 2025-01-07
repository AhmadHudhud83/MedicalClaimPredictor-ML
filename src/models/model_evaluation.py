# Importing Dependencies

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import r2_score,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from dvclive import Live
import yaml


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
def load_model(filepath:str):
    try:
        with open(filepath,"rb") as f:
            print(f"Loaded model from {filepath}")

            return joblib.load(f)
            
    except Exception as e:
        raise Exception(f"Error loading the model from {filepath} : {e}")
    
# Plotting predicted vs actual values for test set
def plot(Y:np.ndarray,data_prediction:pd.Series,set:str,color:str)->str:
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(Y, data_prediction, color=color, alpha=0.1)
        plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', linewidth=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted Values ({set} Set)')
        plot_path = f"reports/figures/{set}_plot.png"
        plt.savefig(plot_path)
        plt.show()
        plt.close()
        return plot_path
    except Exception as e :
        raise Exception(f"Error occured during plotting predicated vs actual values : {e}")


# Model Evaluation function
def evaluation_model(model,X_test: pd.DataFrame,Y_test :pd.Series,X_train: pd.DataFrame,Y_train :pd.Series )->dict:
    try:
        # Loading params
        params = yaml.safe_load(open("params.yaml","r"))
        test_size = params["data_collection"]["test_size"]
        kwargs= params["model_building"]["kwargs"]
        model_name = params["model_building"]["model_name"]

        # Predicting test & train sets
        test_data_prediction = model.predict(X_test)
        train_data_prediction = model.predict(X_train)

        # Defining r square measure
        r2_test = r2_score(Y_test, test_data_prediction)
        r2_train = r2_score(Y_train,train_data_prediction)
        mae = mean_absolute_error(test_data_prediction,Y_test)
       
        # Plotting test & train datasets vs actual values diagrams
        test_plot = plot(Y_test,test_data_prediction,"Test","green")
        train_plot = plot(Y_train,train_data_prediction,"Train","blue")


        # Logging experiment results
        with Live(save_dvc_exp=True) as live:
            live.log_metric("r2_test",r2_test)
            live.log_metric("r2_train",r2_train)
            live.log_metric("MAE",mae)
            live.log_param("test_size",test_size)
            live.log_param("kwargs",kwargs)
            live.log_param("model_name",model_name)
            live.log_image("test_plot",test_plot)
            live.log_image("train_plot",train_plot)

        # Returning metrics dictionary
        metrics_dict= {
                 "r2_test":r2_test,
                 "r2_train":r2_train,
                 "mae":mae
            }
        print(f"Model Evaluation Results: {metrics_dict}")
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
        train_data_path = "./data/interim/train_engineered.csv"
        test_data_path = "./data/interim/test_engineered.csv"
        model_path = "models/model.joblib"
        metrics_path = "reports/metrics.json"
    
        # Loading test & prepare datasets
        test_data =load_data(test_data_path)
        train_data = load_data(train_data_path)
        print("from model evaluation",train_data.info())
        # Defining X-test & Y-test
        X_test,Y_test = prepare_data(test_data)

        # Defining X-train & Y-train
        X_train , Y_train = prepare_data(train_data)


        # Loading Model
        model = load_model(model_path)
       
        
        # Getting metrics based on mode l and test set
        metrics = evaluation_model(model=model,X_test=X_test, Y_test=Y_test,X_train=X_train,Y_train=Y_train)
       
        # # Saving results of metrics
        save_metrics(metrics,metrics_path)
    except Exception as e:
        raise Exception(f"Error Occured  : {e}")

if __name__ == "__main__":
    main()