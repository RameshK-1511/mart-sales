import os
import pandas as pd
import numpy as np
import yaml
from datetime import date
import argparse


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def get_data(config_path):
    try:
        cwd = os.getcwd()
        print(cwd, config_path)
        config = read_params(config_path)
        data_path = config["data_source"]["raw_data"]
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        print("Wrong file or file path")

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="../params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)