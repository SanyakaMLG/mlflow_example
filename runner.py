from scripts import evaluate, process_data, train
import mlflow
from constants import EXPERIMENT_NAME, MLFLOW_URI
import argparse

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Запуск пайплайна машинного обучения")
    parser.add_argument(
        '--run_name', 
        type=str, 
        default='manual_run', 
        help='Имя запуска в MLflow'
    )
    args = parser.parse_args()

    with mlflow.start_run(run_name=args.run_name):
        process_data()
        train()
        evaluate()
