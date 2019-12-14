import json
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from azureml.core.model import Model
from azureml.core import Workspace

def init():
    global model   

    model_name = 'gbr_model_train_msft'
    model_path = Model.get_model_path(model_name)
    model = joblib.load(model_path)


def run(input_json):     
    try:
        inputs = json.loads(input_json)
        prediction = model.predict(inputs)
        prediction = json.dumps(prediction.tolist())

    except Exception as e:
        prediction = str(e)
    return prediction