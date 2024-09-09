# quantum-task-regression
## Regression task

The regression model here is the simple implementation of Linear regression.
It was shown in data_exploratory_analysis notebook, that there is a strong linear
correlation between the 6th squared feature, the 7th feature and target.

#### Tested with Python3.12.0 on MacOS
**Steps to reproduce:**
1. ` python3 -m venv .env `
2. ` source .env/bin/activate `
3. ` python3 -m pip install -r requirements.txt `
> Use the same venv for Jupyter Notebook.

**Retrain the model, use the following command**
> This script saves the .joblib file  into the folder, where the script is executed.

` python3 train.py --dataset_path "/path/to/train.csv" --test_size 0.1  `  


**Predict labels for existing .csv file**
> This scripts saves the new .csv file with all test data and prediction column  

` python3 predict.py --dataset_path "/path/to/hidden_test.csv" --model_path /path/to/model.joblib `  
