import os
import pandas as pd
import numpy as np
import json
from datetime import date


class PreProcess(object):

    def __init__(self):
        self.json_path_file = "params/params_model.json"

    def read_data(self, config):
        config = self.read_params(self.json_path_file)
        train_filename = config["file_path"]["train_path_file"]
        try:
            data = pd.read_csv(train_filename)
            return data
        except Exception as e:
            raise Exception()

    def read_params(self, json_file_path):
        with open(json_file_path) as json_file:
            config = json.load(json_file)

        return config


    def check_missing_values(self, data):
        missing_cols = data.columns[data.isnull().any()].tolist()
        if len(missing_cols) > 0:
            self.impute_miss_cols(data, missing_cols)
        self.clean_data(data)
        print(data.head())
        self.value_mapping(data)
        self.text_replacement(data)
        self.item_visibility(data)
        self.total_established_years(data)
        self.log_transform(data)
        train_data = self.drop_columns(data)

        return train_data

    def clean_data(self, data):
        config = self.read_params(self.json_path_file)
        all_cat_col = config["column"]["categorical_col"]
        for col in all_cat_col:
            data[col] = data[col].str.lower()
            data[col]= data[col].str.replace(' ', '')

    def impute_miss_cols(self, data, missing_cols):
        config = self.read_params(self.json_path_file)

        for col in missing_cols:
            if col in config["column"]["categorical_col"]:
                impute_mode = 'ultra_high'
                data[col] = data[col].fillna(impute_mode)
            elif col in config["column"]["numerical_col"]:
                impute_mean = data[col].mean()
                data[col] = data[col].fillna(impute_mean)

    def value_mapping(self, data):
        config = self.read_params(self.json_path_file)
        col = config["column"]["categorical_col"][0]
        data[col] = data[col].str[:2].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})

    def text_replacement(self, data):
        config = self.read_params(self.json_path_file)
        col = config["column"]["categorical_col"][1]
        data[col] = data[col].replace(['low fat', 'LF', 'reg'], ['Low Fat', 'Low Fat', 'Regular'])

    def log_transform(self, data):
        config = self.read_params(self.json_path_file)
        target = config["column"]["target"][0]
        if target in data.columns.tolist():
            target = np.log(data[target])
            print(target.head())

    def item_visibility(self, data):
        config = self.read_params(self.json_path_file)
        col = config["column"]["numerical_col"][1]
        visibility_median = data[data[col]!=0][col].median()
        data[col] = data[col].replace(0, visibility_median)

    def total_established_years(self, data):
        config = self.read_params(self.json_path_file)
        col = config["column"]["numerical_col"][3]
        data['number_established_years'] = date.today().year - data[col].astype('int')

    def drop_columns(self, data):
        config = self.read_params(self.json_path_file)
        drop_col = config["column"]["numerical_col"][3]
        data = data.drop(columns=[drop_col])

        return data
