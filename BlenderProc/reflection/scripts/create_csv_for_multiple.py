"""
Script to generate a CSV file by scanning a directory tree for .hdf5 files,
extracting UIDs from the directory names, and retrieving corresponding captions
from existing training and testing CSV files for multiple obejcts

Usage:
    python create_csv_for_multiple.py --input_dir /path/to/hdf5/files --output_file output.csv
"""

import glob
import os
from pathlib import Path
import pandas as pd
import argparse

def find_hdf5_files(root_dir):
    """
    Recursively find all .hdf5 files in the given root directory.

    Args:
        root_dir (str): The root directory to search in.

    Returns:
        list of Path: List of relative paths to the found .hdf5 files with respect to root_dir.
    """
    # Find all .hdf5 files in the directory and subdirectories
    file_paths = glob.glob(os.path.join(root_dir, '**', '*.hdf5'), recursive=True)
    # Convert to relative paths
    relative_paths = [Path(file).relative_to(root_dir) for file in file_paths]
    return relative_paths

def main(args):
    """
    Main execution function. Loads metadata, scans the directory for .hdf5 files,
    constructs a test DataFrame, and saves it to a CSV.

    Args:
        args (Namespace): Command-line arguments.
    """
    paths = find_hdf5_files(args.input_dir)
    print(f"Found {len(paths)} HDF5 files.")

    # Load training and test CSV files
    full_train_csv_file_path = "/data/manan/data/objaverse/blenderproc/train.csv"
    full_test_csv_file_path = "/data/manan/data/objaverse/blenderproc/test.csv"

    full_df = pd.read_csv(full_train_csv_file_path)
    full_test_df = pd.read_csv(full_test_csv_file_path)

    column_names = full_df.columns.tolist()

    test_df = pd.DataFrame(columns=column_names)
    
    def get_entry_in_df(query_uid):
        """
        Retrieve a row from either the training or test DataFrame using the UID.

        Args:
            query_uid (str): UID to query.

        Returns:
            pd.DataFrame: A single-row DataFrame matching the UID.
        """
        if (full_df["uid"] == query_uid) is not None:
            return full_df.loc[full_df["uid"] == query_uid].iloc[0:1]
        else:
            return  full_test_df.loc[full_test_df["uid"] == query_uid].iloc[0:1]

    def get_caption(concat_uid):
        """
        Generate a combined caption from two UIDs joined by an underscore.

        Args:
            concat_uid (str): UID string in the format "uid1_uid2".

        Returns:
            str: Combined caption.
        """
        first_uid, second_uid = concat_uid.split("_")
        first_entry = get_entry_in_df(first_uid)
        second_entry =  get_entry_in_df(second_uid)
        return f"{first_entry.iloc[0]['caption']} and {second_entry.iloc[0]['caption']}"

    def get_auto_caption(concat_uid):
        """
        Generate a combined auto_caption from two UIDs joined by an underscore.

        Args:
            concat_uid (str): UID string in the format "uid1_uid2".

        Returns:
            str: Combined auto_caption.
        """
        first_uid, second_uid = concat_uid.split("_")
        first_entry = get_entry_in_df(first_uid)
        second_entry =  get_entry_in_df(second_uid)
        return f"{first_entry.iloc[0]['auto_caption']} and {second_entry.iloc[0]['auto_caption']}"

    # Build test DataFrame
    test_df["path"] = paths
    test_df["uid"] = test_df["path"].apply(lambda x: str(Path(x).parent.name))
    test_df["caption"] = test_df["uid"].apply(get_caption)
    
    # Uncomment below if auto_captions are needed
    # test_df["auto_caption"] = test_df["uid"].apply(get_auto_caption)
    
    test_df["is_novel"] = False

    # Save to output CSV
    test_save_path = os.path.join(args.input_dir, args.output_file)
    test_df.to_csv(test_save_path, index=False)
    print(f"Test CSV saved to {test_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Test CSV file for rotation/novel-views")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data/manan/data/objaverse/blenderproc",
        required=True,
        help="Directory where hdf5 files are generated"
    )
    parser.add_argument(
        "--output_file", type=str, default="test_novel.csv", help="Name of the output csv file"
    )

    args = parser.parse_args()
    main(args)