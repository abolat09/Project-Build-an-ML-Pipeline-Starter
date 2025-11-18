"""
This script downloads a dataset, performs basic cleaning,
and logs the cleaned dataset as a W&B artifact.
"""
import argparse
import logging
import pandas as pd
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def go(args):
    """
    Main function to execute the data cleaning and logging process.

    :param args: argparse.Namespace, containing the script's command-line arguments.
    """
    run = wandb.init(job_type="basic_cleaning")

    logging.info("Downloading artifact: %s", args.input_artifact)
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    logging.info("Loading artifact into pandas DataFrame...")
    df = pd.read_csv(artifact_path)

    logging.info("Dropping rows with missing values...")
    df.dropna(subset=[args.col_to_clean1, args.col_to_clean2], inplace=True)

    logging.info("Filtering by price range: %s to %s", args.min_price, args.max_price)
    df = df[(df['price'] >= args.min_price) & (df['price'] <= args.max_price)]
    
    #logging.info("Filtering by geographic boundaries...")
    #df = df[
        #(df['longitude'] >= -74.25) & (df['longitude'] <= -73.50) &
        #(df['latitude'] >= 40.5) & (df['latitude'] <= 41.2)
    #]

    df.to_csv("clean_data.csv", index=False)

    logging.info("Logging cleaned data artifact: %s", args.output_artifact)
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_data.csv")
    run.log_artifact(artifact)

    logging.info("Basic cleaning step finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform basic cleaning on the raw data.")

    parser.add_argument(
        "--input_artifact", 
        type=str, 
        help="Name of the input artifact (raw data)", 
        required=True
    )
    parser.add_argument(
        "--output_artifact", 
        type=str, 
        help="Name for the output artifact (cleaned data)", 
        required=True
    )
    parser.add_argument(
        "--output_type", 
        type=str, 
        help="Type of the output artifact", 
        required=True
    )
    parser.add_argument(
        "--output_description", 
        type=str, 
        help="Description for the output artifact", 
        required=True
    )
    parser.add_argument(
        "--col_to_clean1", 
        type=str, 
        help="First column to drop NaNs from", 
        required=True
    )
    parser.add_argument(
        "--col_to_clean2", 
        type=str, 
        help="Second column to drop NaNs from", 
        required=True
    )
    parser.add_argument(
        "--min_price", 
        type=float, 
        help="Minimum price to keep", 
        required=True
    )
    parser.add_argument(
        "--max_price", 
        type=float, 
        help="Maximum price to keep", 
        required=True
    )

    args = parser.parse_args()
    go(args)