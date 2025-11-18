"""
This script performs data checks on the cleaned dataset.
It combines pre-written tests with new tests for row count and price range.
"""
import pandas as pd
import numpy as np
import scipy.stats
import argparse
import logging
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Existing Tests (from your file) ---

def test_column_names(data: pd.DataFrame) -> None:
    """Test if the DataFrame has the expected column names.
    
    Args:
        data: Input DataFrame to test
    """
    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]
    these_columns = data.columns.values
    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)
    logging.info("Column names test passed.")

def test_neighborhood_names(data: pd.DataFrame) -> None:
    """Test if neighborhood names are within expected values.
    
    Args:
        data: Input DataFrame to test
    """
    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    neigh = set(data['neighbourhood_group'].unique())
    # Unordered check
    assert set(known_names) == set(neigh)
    logging.info("Neighborhood names test passed.")

def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)
    assert np.sum(~idx) == 0
    logging.info("Proper boundaries test passed.")

def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float) -> None:
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    
    Args:
        data: Current dataset to test
        ref_data: Reference dataset to compare against
        kl_threshold: Maximum allowed KL divergence threshold
        
    Raises:
        AssertionError: If KL divergence exceeds the threshold
    """
    dist1 = data['neighbourhood_group'].value_counts(normalize=True).sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts(normalize=True).sort_index()
    
    assert np.isclose(dist1.sum(), 1.0)
    assert np.isclose(dist2.sum(), 1.0)
    assert dist1.index.equals(dist2.index)

    kl_div = scipy.stats.entropy(dist1, dist2, base=2)
    assert np.isfinite(kl_div) and kl_div < kl_threshold
    logging.info("Neighborhood distribution test passed.")

# --- New Tests (from rubric) ---

def test_row_count(data: pd.DataFrame):
    """
    Tests if the row count is within a reasonable range.
    """
    logging.info("Testing row count...")
    assert 15000 < data.shape[0] < 1000000
    logging.info("Row count test passed.")

def test_price_range(data: pd.DataFrame, min_price: float, max_price: float):
    """
    Tests if the price column is within the expected range.
    """
    logging.info("Testing price range...")
    assert data['price'].between(min_price, max_price).all()
    logging.info("Price range test passed.")

# --- Main execution logic ---

def go(args):
    """
    Main function to run the data checks.
    """
    run = wandb.init(job_type="data_check")

    logging.info("Downloading artifact: %s", args.csv)
    artifact = run.use_artifact(args.csv)
    artifact_path = artifact.file()
    data = pd.read_csv(artifact_path)

    logging.info("Downloading reference artifact: %s", args.ref)
    ref_artifact = run.use_artifact(args.ref)
    ref_artifact_path = ref_artifact.file()
    ref_data = pd.read_csv(ref_artifact_path)

    # Run all tests
    logging.info("Running data tests...")
    test_column_names(data)
    test_neighborhood_names(data)
    test_proper_boundaries(data)
    test_similar_neigh_distrib(data, ref_data, args.kl_threshold)
    test_row_count(data)
    test_price_range(data, args.min_price, args.max_price)
    
    logging.info("All data tests passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data checks on the cleaned data.")

    parser.add_argument(
        "--csv",
        type=str,
        help="Name of the input artifact (cleaned data)",
        required=True
    )
    parser.add_argument(
        "--ref",
        type=str,
        help="Name of the reference artifact",
        required=True
    )
    parser.add_argument(
        "--kl_threshold",
        type=float,
        help="KL divergence threshold",
        required=True
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum expected price",
        required=True
    )
    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum expected price",
        required=True
    )

    args = parser.parse_args()
    go(args)