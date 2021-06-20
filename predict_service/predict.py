import yaml
import os
import json
import joblib
import pickle
import collections
import numpy as np
from scipy.sparse import data, hstack

from src.get_data import read_params, get_data
from src.load_data import PreProcess


params_path = "params.yaml"
schema_path = os.path.join("predict_service", "schema_input.json")

class NotInRange(Exception):
    def __init__(self, message="Values entered are not in expected range"):
        self.message = message
        super().__init__(self.message)

class NotInCols(Exception):
    def __init__(self, message="Not in cols"):
        self.message = message
        super().__init__(self.message)


def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predict(data):
    config = read_params(params_path)
    model_path = config["predict_model_dir"]

    predict_data = PreProcess()
    processed_data = predict_data.missing_values(data)

    categorical_cols = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
    numerical_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'number_established_years']

    standardize_num_data = processed_data[numerical_cols]

    std_pkl_path = os.path.join(model_path, "standard_scaler.pkl")
    print('standardize_num_data', standardize_num_data)
    sc = pickle.load(open(std_pkl_path, 'rb'))
    standardize_data = sc.transform(standardize_num_data)

    vectorize_cat_data = processed_data[categorical_cols]
    vec_pkl_path = os.path.join(model_path, "label_encode.pkl")
    le = pickle.load(open(vec_pkl_path, 'rb'))
    vectorize_data = le.transform(vectorize_cat_data)

    predict_data = hstack((standardize_data, vectorize_data)).tocsr()

    model_pkl_path = os.path.join(model_path, "rf_model.pkl")
    model = pickle.load(open(model_pkl_path, 'rb'))
    prediction = model.predict(predict_data)[0]

    print(prediction)
    try:
        if prediction > 0:
            return prediction
        else:
            raise NotInRange
    except NotInRange:
        return "Unexpected result"

    return prediction


def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema


def validate_input(data):
    def _validate_cols(cols):
        schema = get_schema()
        actual_cols = list(schema.keys())
        if collections.Counter(cols) != collections.Counter(actual_cols):
            raise NotInCols

    cols = data.columns.tolist()
    _validate_cols(cols)
    
    return True


def form_response(data):
    if validate_input(data):
        response = predict(data)
        return response


def api_response(data):
    print(data)
    try:
        if validate_input(data):
            response = predict(data)
            response = {"response": response}
            return response
            
    except NotInRange as e:
        response = {"the_exected_range": get_schema(), "response": str(e) }
        return response

    except NotInCols as e:
        response = {"the_exected_cols": get_schema().keys(), "response": str(e) }
        return response

    except Exception as e:
        response = {"response": str(e) }
        return response