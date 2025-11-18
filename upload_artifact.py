import wandb
import argparse
import os

def upload_file(file_path, project_name, artifact_name, artifact_type):
    """
    Logs a local file as a new W&B artifact.
    """
    run = wandb.init(project=project_name, job_type="upload_artifact")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    print(f"Creating artifact '{artifact_name}'...")
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type
    )

    print(f"Adding file '{file_path}' to artifact...")
    artifact.add_file(file_path)

    print("Logging artifact to W&B...")
    run.log_artifact(artifact)

    print("Done.")
    run.finish()

if __name__ == "__main__":
    # Make sure sample2.csv is in the same directory as this script!
    upload_file(
        file_path="sample2.csv",
        project_name="nyc_airbnb",
        artifact_name="sample2.csv",
        artifact_type="raw_data"
    )