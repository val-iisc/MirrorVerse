import glob
import os
from pathlib import Path
import pandas as pd
import argparse
import random
random.seed(10)

def find_hdf5_files(root_dir):
    # Find all .hdf5 files in the directory and subdirectories
    file_paths = glob.glob(os.path.join(root_dir, '**', '*.hdf5'), recursive=True)
    # Convert to relative paths
    relative_paths = [Path(file).relative_to(root_dir) for file in file_paths]
    return relative_paths

def create_train_test_split(paths, train_ratio=0.95):
    random.shuffle(paths)
    split_idx = int(train_ratio * len(paths))
    train_paths = paths[:split_idx]
    test_paths = paths[split_idx:]
    return train_paths, test_paths

def create_csv(list_paths, args, file_name):
    #Create data-frame
    df = pd.DataFrame(list_paths, columns=["path"])
    df["uid"] = df["path"].apply(lambda x: Path(x).parent.name)

    # import captions csv
    captions = pd.read_csv(args.caption_path, header=None, names=["uid", "caption"])

    # Merge test DataFrame with captions DataFrame on 'uid' column
    merged_df = pd.merge(df, captions, on="uid")

    save_path = os.path.join(args.input_dir, file_name)
    merged_df.to_csv(save_path, index=False)

def main(args):
    paths = find_hdf5_files(args.input_dir)
    print(f"Found {len(paths)} HDF5 files.")

    #create train-test split
    train_paths, test_paths = create_train_test_split(paths)
    print(f"No. of train data-points: {len(train_paths)}")
    print(f"No. of test data-points: {len(test_paths)}")

    create_csv(train_paths, args, "train_abo.csv")
    create_csv(test_paths, args, "test_abo.csv")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Test CSV file")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory where hdf5 files are generated"
    )

    parser.add_argument(
        "--caption_path", 
        type=str, 
        default="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/ankit/captions/Cap3D/misc/Cap3D_automated_ABO.csv", 
        help="path where all captions are saved."
    )

    args = parser.parse_args()
    main(args)