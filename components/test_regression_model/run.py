import argparse
import logging
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def go(args):
    run = wandb.init(job_type="test_model")

    logging.info(f"Downloading artifact: {args.mlflow_model}")
    model_artifact = run.use_artifact(args.mlflow_model)
    model_dir = model_artifact.download()

    logging.info(f"Downloading artifact: {args.test_dataset}")
    test_artifact = run.use_artifact(args.test_dataset)
    test_path = test_artifact.file()

    logging.info("Loading model and test data...")
    model = mlflow.sklearn.load_model(model_dir)
    df = pd.read_csv(test_path)

    X_test = df.drop(columns="price")
    y_test = df["price"]

    logging.info("Running inference...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    logging.info(f"Test MAE: {mae}")
    run.summary["test_mae"] = mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_model", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    args = parser.parse_args()
    go(args)