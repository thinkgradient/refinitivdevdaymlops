import pandas as pd
import numpy as np
import os
import argparse
import statsmodels.api as sm
from azureml.core.run import Run
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from interpret.ext.blackbox import TabularExplainer
from azureml.contrib.interpret.explanation.explanation_client import ExplanationClient


OUTPUT_DIR = './outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Retrieve command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str,  help='data folder mounting point')
parser.add_argument('--filename', type=str,  help='training file name')
args = parser.parse_args()

# Configure a path to training data
# data_folder = os.path.join(args.data_folder, 'datasets')
# print('Loading data from: ', data_folder)
# data_csv_path = os.path.join(data_folder, args.filename)
data_csv_path = args.data_folder + '/' + args.filename


# Load the dataset
df = pd.read_csv(data_csv_path)


targets = df['Close']
features = df.drop(['Close'], axis=1)

linear_features = sm.add_constant(features)
linear_features.columns

# Create a size for the training set that is 85% of the total number of samples
train_size = int(0.85 * targets.shape[0])
train_features = linear_features[:train_size]
train_targets = targets[:train_size]
test_features = linear_features[train_size:]
test_targets = targets[train_size:]
print(linear_features.shape, train_features.shape, test_features.shape)


#try out gradient boosting model
from sklearn.ensemble import GradientBoostingRegressor

# Create GB model -- hyperparameters have already been searched for you
gbr = GradientBoostingRegressor(max_features=4,
                                learning_rate=0.01,
                                n_estimators=200,
                                subsample=0.6,
                                random_state=42)
gbr.fit(train_features, train_targets)


train_r2 = gbr.score(train_features, train_targets)
test_r2 = gbr.score(test_features, test_targets)
preds = gbr.predict(test_features)

rmse = np.sqrt(mean_squared_error(test_targets, preds))




run = Run.get_context()
client = ExplanationClient.from_run(run)

run.log("max_features", 4)
run.log("learning_rate", 0.01)
run.log("n_estimators", 200)
run.log("subsample", 0.6)
run.log("random_state", 42)
run.log("Train R2 Score", train_r2)
run.log("Test R2 Score", test_r2)
run.log("RMSE", rmse)

print("Saving the model to outputs ...")

model_file_name = 'gbr_tickfund.pkl'
joblib.dump(value=gbr, filename='outputs/model.pkl')

with open(model_file_name, 'wb') as file:
    joblib.dump(value=gbr, filename=os.path.join(OUTPUT_DIR,
                                                 model_file_name))
# register the model
run.upload_file('dev_model.pkl', os.path.join('./outputs/', model_file_name))
original_model = run.register_model(model_name='gbr_model_train_msft',
                                    model_path='dev_model.pkl')

# Explain predictions on your local machine
tabular_explainer = TabularExplainer(gbr, train_features, features=df.columns)

# Explain overall model predictions (global explanation)
# Passing in test dataset for evaluation examples - note it must be a representative sample of the original data
# x_train can be passed as well, but with more examples explanations it will
# take longer although they may be more accurate
global_explanation = tabular_explainer.explain_global(test_features)

# Uploading model explanation data for storage or visualization in webUX
# The explanation can then be downloaded on any compute
comment = 'Global explanation on regression model trained on ticker fund dataset'
client.upload_model_explanation(global_explanation, comment=comment, model_id=original_model.id)


