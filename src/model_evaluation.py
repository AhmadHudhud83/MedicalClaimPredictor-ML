# Importing Dependencies
import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Reading test data
test_data =pd.read_csv("./data/engineered/test_engineered.csv")

X_test = test_data.iloc[:, 1:].values
Y_test = test_data.iloc[:, 0].values
model = pickle.load(open("model.pkl","rb"))
test_data_prediction = model.predict(X_test)
r2_test = r2_score(Y_test, test_data_prediction)
print(X_test.shape)
print('R squared value (Test): ', r2_test)

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, test_data_prediction, color='green', alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Test Set)')
plt.show()

metrics_dict= {
    "r2_test":r2_test
}
with open('metrics.json','w') as f:
    json.dump(metrics_dict,f,indent=4)