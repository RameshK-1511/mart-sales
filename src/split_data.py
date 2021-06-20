import os
import argparse
import pickle
import pandas as pd
from scipy.sparse import data
from get_data import read_params
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Features:

    def split_and_saved_data(self, config_path):
        config = read_params(config_path)
        test_data_path = config["split_data"]["test_path"] 
        train_data_path = config["split_data"]["train_path"]
        raw_data_path = config["load_data"]["raw_dataset_csv"]
        test_size = config["split_data"]["test_size"]
        random_state = config["base"]["random_state"]
        model_dir = config["model_dir"]

        dataset = pd.read_csv(raw_data_path)

        train, test = train_test_split(dataset, test_size=test_size, random_state=random_state)
        train.to_csv(train_data_path, sep=",", index=False)
        test.to_csv(test_data_path, sep=",", index=False)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="../params.yaml")
    parsed_args = args.parse_args()
    fe = Features()
    fe.split_and_saved_data(config_path=parsed_args.config)