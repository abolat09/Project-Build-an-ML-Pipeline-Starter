import argparse
import logging
import wandb
import os
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def go(args):
    run = wandb.init(job_type="download")

    logging.info(f"Downloading W&B artifact: {args.sample}")
    try:
        artifact = run.use_artifact(args.sample)
        download_dir = artifact.download("./artifact_download")

        downloaded_files = os.listdir(download_dir)
        if not downloaded_files:
            raise FileNotFoundError(f"No files found in downloaded artifact: {args.sample}")

        source_path = os.path.join(download_dir, downloaded_files[0])
        target_path = args.artifact_name

        shutil.move(source_path, target_path)
        final_download_path = target_path

    except Exception as e:
        logging.error(f"Failed to download W&B artifact: {e}")
        raise e

    logging.info(f"Logging artifact {args.artifact_name} to W&B")
    output_artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )
    output_artifact.add_file(final_download_path)
    run.log_artifact(output_artifact)
    logging.info("Download step finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--artifact_name", type=str, required=True)
    parser.add_argument("--artifact_type", type=str, required=True)
    parser.add_argument("--artifact_description", type=str, required=True)

    args = parser.parse_args()
    go(args)