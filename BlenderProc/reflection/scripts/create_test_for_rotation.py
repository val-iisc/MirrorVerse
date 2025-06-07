import glob
import os
from pathlib import Path
import pandas as pd
import argparse

def find_hdf5_files(root_dir):
    # Find all .hdf5 files in the directory and subdirectories
    file_paths = glob.glob(os.path.join(root_dir, '**', '*.hdf5'), recursive=True)
    # Convert to relative paths
    relative_paths = [Path(file).relative_to(root_dir) for file in file_paths]
    return relative_paths

def main(args):
    paths = find_hdf5_files(args.input_dir)
    print(f"Found {len(paths)} HDF5 files.")

    #Read test-csv file
    full_test_csv_file_path = "/data/manan/data/objaverse/blenderproc/test.csv"

    full_df = pd.read_csv(full_test_csv_file_path)

    column_names = full_df.columns.tolist()

    test_df = pd.DataFrame(columns=column_names)
    print(test_df)

    test_df["path"] = paths
    test_df["uid"] = test_df["path"].apply(lambda x: str(Path(x).parent.name))
    test_df["orig_uid"] = test_df["path"].apply(lambda x: str(Path(x).parent.name).split("_")[0])
    test_df["caption"] = test_df["orig_uid"].apply(lambda x: full_df.loc[full_df["uid"] == x].iloc[0:1]["caption"])
    test_df["auto_caption"] = test_df["orig_uid"].apply(lambda x: full_df.loc[full_df["uid"] == x].iloc[0:1]["auto_caption"])
    test_df["is_novel"] = test_df["orig_uid"].apply(lambda x: full_df.loc[full_df["uid"] == x].iloc[0:1]["is_novel"])

    test_df = test_df.drop(columns=["orig_uid"])
    test_save_path = os.path.join(args.input_dir, args.output_file)
    test_df.to_csv(test_save_path, index=False)


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