import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'train'


def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    model_type = params.get('model_type', 'LogisticRegression')
    model_params = params.get('model_params', {})
    model_params['random_state'] = RANDOM_STATE

    logger.info(f'Создаём модель: {model_type}')
    logger.info(f'    Параметры модели: {model_params}')

    mlflow.log_param("model_type", model_type)
    for key, value in model_params.items():
        mlflow.log_param(f"model_param_{key}", value)

    if model_type == 'LogisticRegression':
        model = LogisticRegression(**model_params)
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(**model_params)
    elif model_type == 'XGBoost':
        model = XGBClassifier(**model_params)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    logger.info('Обучаем модель')
    model.fit(X_train, y_train)

    logger.info('Сохраняем модель')
    dump(model, MODEL_FILEPATH)
    mlflow.sklearn.log_model(model, "model")
    logger.info('Успешно!')


if __name__ == '__main__':
    train()
