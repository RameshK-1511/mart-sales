import os
import yaml
import argparse
import pandas as pd
# from src.predict import Prediction
from src.preprocess import PreProcess
from src.train_model import TrainModel


# def predict(predict_file_path):
#     test_pre = Prediction()
#     test_data = test_pre.read_data(predict_file_path)
#     test_pre.model_prediction(test_data, csv_flg='Y')


def train(train_file_path):
    pre = PreProcess()
    data = pre.read_data(train_file_path)
    processed_data = pre.check_missing_values(data)
    tm = TrainModel()
    tm.train_test_split(processed_data)

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    data_path = config["data_source"]["s3_source"]
    return data_path

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params/params.yaml")
    parsed_args = args.parse_args()
    data_path = get_data(config_path=parsed_args.config)
    print(data_path)
    train(data_path)
    # predict(test_file_path)
