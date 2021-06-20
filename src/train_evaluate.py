import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.sparse import hstack
from get_data import read_params
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

class TrainEvaluate():

    def train_and_evaluate(self, config_path):
        config = read_params(config_path)
        test_data_path = config["split_data"]["test_path"]
        train_data_path = config["split_data"]["train_path"]
        random_state = config["base"]["random_state"]
        target = config["base"]["target_col"]
        model_dir = config["model_dir"]

        train = pd.read_csv(train_data_path)
        test = pd.read_csv(test_data_path)

        X_train = train.loc[ : , train.columns != target]
        y_train = train[target]

        X_test = test.loc[ : , test.columns != target]
        y_test = test[target]

        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X_train.select_dtypes(include=['int','float']).columns.tolist()

        print('categorical_cols', categorical_cols)
        print('numerical_cols', numerical_cols)

        std_train_data, std_test_data = self.standardize(X_train, X_test, numerical_cols, model_dir)
        vect_train_data, vect_test_data = self.one_hot_encoder(X_train, X_test, categorical_cols, model_dir)
        self.build_model(std_train_data, std_test_data, vect_train_data, vect_test_data, y_train, y_test, config)

    def standardize(self, train_data, test_data, standardize_columns, model_dir):
        # standardizing numerical features such that they have mean = 0, sd = 1
        scaler = StandardScaler()
        train_data = train_data[standardize_columns]
        test_data = test_data[standardize_columns]

        scaled_data = scaler.fit(train_data)
        std_train_data = scaled_data.transform(train_data)
        std_test_data = scaled_data.transform(test_data)

        os.makedirs(model_dir, exist_ok=True)
        pkl_path = os.path.join(model_dir, "standard_scaler.pkl")

        with open(pkl_path, 'wb') as f:
            pickle.dump(scaled_data, f)

        return std_train_data, std_test_data


    def one_hot_encoder(self, train_data, test_data, vectorize_columns,model_dir):
        # One hot encoding which gives bag of words representation for categorical features
        train_data = train_data[vectorize_columns]
        test_data = test_data[vectorize_columns]

        le = OneHotEncoder()
        encode_data = le.fit(train_data)

        vect_train_data = encode_data.transform(train_data)
        vect_test_data = encode_data.transform(test_data)
        
        os.makedirs(model_dir, exist_ok=True)
        pkl_path = os.path.join(model_dir, "label_encode.pkl")

        with open(pkl_path, 'wb') as f:
            pickle.dump(encode_data, f)

        return vect_train_data, vect_test_data


    def build_model(self, standard_train, standard_test, vectorize_train, vectorize_test, y_train, y_test, config):
        # stack features for training
        x_train = hstack((standard_train, vectorize_train)).tocsr()
        x_test = hstack((standard_test, vectorize_test)).tocsr()

        self.model_train_rf(x_train, y_train, x_test, y_test, config)


    def model_train_rf(self, x_train, y_train, x_test, y_test, config):

        print('*'*200)
        print('RANDOM FOREST REGRESSOR')
        print('*'*200)

        n_estimators = config["estimators"]["RandomForest"]["params"]["n_estimators"]
        max_depth = config["estimators"]["RandomForest"]["params"]["max_depth"]
        max_features = config["estimators"]["RandomForest"]["params"]["max_features"]
        min_samples_split = config["estimators"]["RandomForest"]["params"]["min_samples_split"]
        min_samples_leaf = config["estimators"]["RandomForest"]["params"]["min_samples_leaf"]
        model_dir = config["model_dir"]
        scores_file = config["reports"]["scores"]
        params_file = config["reports"]["params"]

        print(n_estimators)

        # Create the random grid 'n_estimators': n_estimators,
        random_grid = { 'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf}

        # print(random_grid)

        rf = RandomForestRegressor()

        # Random search of parameters, using 5 fold cross validation,
        # search across 50 different combinations
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                       scoring='neg_mean_squared_error', cv=5, verbose=12, random_state=42, n_jobs=-1)

        rf_random.fit(x_train, y_train.values.ravel())

        best_param = rf_random.best_params_

        print(best_param)
        print(rf_random.best_score_)

        y_predicted = rf_random.predict(x_test)

        rmse, mae, r2 = self.eval_metrics(y_test, y_predicted)

        eval_metric = {'rmse':rmse, 'mae':mae, 'r2':r2}

        self.report(eval_metric, scores_file, params_file, best_param)

        os.makedirs(model_dir, exist_ok=True)
        pkl_path = os.path.join(model_dir, "rf_model.pkl")

        with open(pkl_path, 'wb') as f:
            pickle.dump(rf_random, f)

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(metrics.mean_squared_error(actual, pred))
        mae = metrics.mean_absolute_error(actual, pred)
        r2 = metrics.r2_score(actual, pred)
        return rmse, mae, r2


    def report(self, eval_metric, scores_file, params_file, best_param):

        with open(scores_file, "w") as f:
            scores = eval_metric
            json.dump(scores, f, indent=4)

        with open(params_file, "w") as f:
            params = best_param
            json.dump(params, f, indent=4)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="../params.yaml")
    parsed_args = args.parse_args()
    te = TrainEvaluate()
    te.train_and_evaluate(config_path=parsed_args.config)