import argparse
from typing import Any

import pandas as pd
import wandb


def go(
    input_artifact: str,
    output_artifact: str,
    output_type: str,
    output_description: str,
    min_price: float,
    max_price: float,
) -> None:
    """
    This is the basic cleaning step.
    1.) We download the input CSV from W&B
    2.) We keep only rows with price between min_price and max_price
    3.) Finally we upload the cleaned CSV back to W&B as a new artifact
    """

    run = wandb.init(
    project="nyc_airbnb",
    job_type="basic_cleaning")


    run.config["input_artifact"] = input_artifact
    run.config["output_artifact"] = output_artifact
    run.config["min_price"] = min_price
    run.config["max_price"] = max_price

    artifact = run.use_artifact(input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)

    df = df[df["price"].between(min_price, max_price)].copy()

    # Here I'm saving the data
    df.to_csv(output_artifact, index=False)

    # Now I will log cleaned artifact to W&B
    cleaned = wandb.Artifact(
        name=output_artifact,
        type=output_type,
        description=output_description,
    )
    cleaned.add_file(output_artifact)
    run.log_artifact(cleaned)

    run.finish()


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="Basic data cleaning")

    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)
    parser.add_argument("--output_type", type=str, required=True)
    parser.add_argument("--output_description", type=str, required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    go(
        input_artifact=args.input_artifact,
        output_artifact=args.output_artifact,
        output_type=args.output_type,
        output_description=args.output_description,
        min_price=args.min_price,
        max_price=args.max_price,
    )
