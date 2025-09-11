from logging import exception
import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pymysql

import pickle
import numpy as np

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv('db')


def read_seq_data():
    logging.info("Reading sql database started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection established", mydb)
        df = pd.read_sql_query("Select * from students", mydb)
        print(df.head())

        return df

    except Exception as ex:
        raise CustomException(ex)


def save_object(file_path, obj):     # <--- Unindented to module level
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        model_names = list(models.keys())
        for i in range(len(model_names)):
            model_name = model_names[i]
            model = models[model_name]
            parameters = param.get(model_name, {})

            gs = GridSearchCV(model, parameters, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
