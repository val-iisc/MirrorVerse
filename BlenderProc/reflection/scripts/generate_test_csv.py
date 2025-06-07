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
    captions_path = args.caption_path
    paths = find_hdf5_files(args.input_dir)
    print(f"Found {len(paths)} HDF5 files.")

    #Create data-frame
    test_df = pd.DataFrame(paths, columns=["path"])
    test_df["uid"] = test_df["path"].apply(lambda x: Path(x).parent.name)

    # import captions csv
    captions = pd.read_csv(captions_path, header=None, names=["uid", "caption"])

    # Merge test DataFrame with captions DataFrame on 'uid' column
    test_merged_df = pd.merge(test_df, captions, on="uid")

    test_save_path = os.path.join(args.input_dir, args.output_file)
    test_merged_df.to_csv(test_save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Test CSV file")
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
    parser.add_argument(
        "--caption_path", 
        type=str, 
        default="/home/test/ankit/data/Cap3D_automated_Objaverse_full.csv", 
        help="path where all captions are saved."
    )

    args = parser.parse_args()
    main(args)