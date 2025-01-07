# Importing Dependencies
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml



#Loading the data from csv file to a Pandas DataFrame
def load_data(filepath: str,batch_size:int)->pd.DataFrame:
    try:
     return pd.read_csv(filepath).head(batch_size)
    
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} : {e}")


#Load params function
def load_params(filepath : str):
    try:
        with open(filepath,"r") as f:
            params = yaml.safe_load(f)
        return params
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}:{e}")




#Splitting Data into Training Set & Testing Set
def split_data(data : pd.DataFrame,test_size:float)->tuple[pd.DataFrame,pd.DataFrame]:
    try:
        return train_test_split(data,test_size=test_size,random_state=101)
    
    except ValueError as e:
        raise ValueError(f"Error Splitting data : {e}")



# Saving data function as an output for this stage
def save_data(df: pd.DataFrame,filepath:str)->None:
    try:
     df.to_csv(filepath,index=False)
     
    except Exception as e:
        raise Exception(f"Error saving data to {filepath} : {e}")


def main():

    try:
        # Setting up filepaths
        data_filepath = r"C:\Users\Gigabyte\Desktop\medicalmalpractice.csv"
        params_filepath = "params.yaml"
        raw_data_path = os.path.join("data","raw")

        #Loading dataset , test_size parameter , training & testing sets
        
        params = load_params(params_filepath)
        test_size = params["data_collection"]["test_size"]
        batch_size = params["data_collection"]["batch_size"]
        dataset = load_data(data_filepath,batch_size)
        train_data , test_data = split_data(dataset,test_size)
        

        #Create data folder 

        os.makedirs(raw_data_path,exist_ok=True)

        #Storing training & testing sets as csv files
        
        save_data(train_data,os.path.join(raw_data_path,"train.csv"))
        save_data(test_data,os.path.join(raw_data_path,"test.csv"))

    except Exception as e:
        raise Exception(f"An error occured : {e}")
    
# Calling main function
if __name__ == "__main__":
    main()