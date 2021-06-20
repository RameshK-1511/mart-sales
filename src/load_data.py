import os
from src.get_data import read_params, get_data
import argparse
import numpy as np
from datetime import date

class PreProcess():
# Check any missing values and treat them via mean or mode imputation
    def load_and_save(self, config_path):
        config = read_params(config_path)
        data = get_data(config_path)

        raw_data_path = config["load_data"]["raw_dataset_csv"]
        print('*'*200)
        print('Ram')
        # print(data)
        print('*'*200)

        data = self.missing_values(data)
        # self.value_mapping(data)
        # self.clean_data(data)
        # self.text_replacement(data)
        # self.item_visibility(data)
        # self.total_established_years(data)
        # self.log_transform(data)
        # self.drop_columns(data)
        # print(data.columns)
        print(data.head())
        data.to_csv(raw_data_path, sep=",", index=False)

# If any missing values in dataset impute them by mean or mode method
    def missing_values(self, data):
        missing_cols = data.columns[data.isnull().any()].tolist()
        if len(missing_cols) > 0:
            self.impute_miss_cols(data, missing_cols)
        self.value_mapping(data)
        self.clean_data(data)
        self.text_replacement(data)
        self.item_visibility(data)
        self.total_established_years(data)
        self.log_transform(data)
        self.drop_columns(data)
        print(data.columns)
        return data
        # return df


    def impute_miss_cols(self, data, missing_cols):
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = data.select_dtypes(include=['int','float']).columns.tolist()
        for col in missing_cols:
            if col in categorical_cols:
                impute_mode = 'ultra_high'
                data[col] = data[col].fillna(impute_mode)
            elif col in numerical_cols:
                impute_mean = data[col].mean()
                data[col] = data[col].fillna(impute_mean)

# Remove any spaces from dataset
    def clean_data(self, data):
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            data[col] = data[col].str.lower()
            data[col]= data[col].str.replace(' ', '_')
        

# Mapping Item_Identifier
    def value_mapping(self, data):
        col = 'Item_Identifier'
        data[col] = data[col].str[:2].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})

    def text_replacement(self, data):
        col = 'Item_Fat_Content'
        data[col] = data[col].replace(['low fat', 'LF', 'reg'], ['Low Fat', 'Low Fat', 'Regular'])


# target feature has some skewness so applying log-transform
    def log_transform(self, data):
        target = 'Item_Outlet_Sales'
        if target in data.columns.tolist():
            target = np.log(data[target])


# Few rows have '0' visibility so imputing them with median
    def item_visibility(self, data):
        col = 'Item_Weight'
        visibility_median = data[data[col]!=0][col].median()
        data[col] = data[col].replace(0, visibility_median)


# Calculating number of years from estabished year
    def total_established_years(self, data):
        col = 'Outlet_Establishment_Year'
        data['number_established_years'] = date.today().year - data[col].astype('int')

# Drop columns
    def drop_columns(self, data):
        drop_col = 'Outlet_Establishment_Year'
        data = data.drop(columns=[drop_col], inplace=True)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="../params.yaml")
    parsed_args = args.parse_args()
    pp = PreProcess()
    pp.load_and_save(config_path=parsed_args.config)