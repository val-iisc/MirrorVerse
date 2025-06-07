"""This is for creating splits of the dataset for rendering on multiple machines."""
import os
import argparse
from loguru import logger as log
import sys
import glob

def find_files(input_dir):
    if not os.path.exists(input_dir):
        log.error(f"Input Directory does not exist. Exit. {input_dir}.")
        sys.exit(-1)
    
    if not os.path.isdir(input_dir):
        log.error(f"Not a directory. Exit. {input_dir}.")
        sys.exit(-1)

    glb_files = glob.glob( os.path.join(input_dir, "**/*.glb") )

    log.info(f"Found {len(glb_files)} glb files.")

    #Extract uids.
    glb_files = [ os.path.split(g_file)[1].split('.')[0]  for g_file in glb_files ]

    #Print some sample glb files
    log.info(f"found uids :\n {glb_files[0:5]} ")

    return glb_files

def collect_uids(base_dir):
    uids_no_small_mirrors = set()
    uids_small_mirrors = set()

    # Traverse the base directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".hdf5"):
                # Extract the uid (the immediate parent folder of the hdf5 file)
                uid = os.path.basename(root)

                # Check if 'small_mirrors' is in the path
                if "small_mirrors" in root:
                    uids_small_mirrors.add(uid)
                else:
                    uids_no_small_mirrors.add(uid)

    return uids_no_small_mirrors, uids_small_mirrors


def create_splits(args, uids, file_prefix="large"):

    uids = list(uids)
    num_samples = len(uids)
    num_samples_per_split = num_samples // args.num_splits

    for i in range(args.num_splits):
        split_uids = uids[i * num_samples_per_split : (i + 1) * num_samples_per_split]
        split_path = os.path.join(args.output_dir, f"{file_prefix}_split_{i}.txt")
        with open(split_path, "w") as f:
            for uid in split_uids:
                f.write(f"{uid}\n")
        log.info(f"Saved split {i} to {split_path}.")


def main(args):
    
    uids_no_small_mirrors, uids_small_mirrors = collect_uids(args.rendered_dir)
    log.info(f'no. of large mirror uids: {len(uids_no_small_mirrors)}, small mirror uids: {len(uids_small_mirrors)}')
    os.makedirs(args.output_dir, exist_ok=True)
    # output uids to text file
    with open(os.path.join(args.output_dir, "large_mirror_uids.txt"), "w") as f:
        for uid in uids_no_small_mirrors:
            f.write(f"{uid}\n")
    with open(os.path.join(args.output_dir, "small_mirror_uids.txt"), "w") as f:
        for uid in uids_small_mirrors:
            f.write(f"{uid}\n")

    create_splits(args, uids_no_small_mirrors, "large")
    create_splits(args, uids_small_mirrors, "small")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create splits of the dataset for rendering on multiple machines.")
    parser.add_argument("--rendered_dir", type=str, default="", help="Path to the previously rendered directory.")
    parser.add_argument("--num_splits", type=int, default=3, help="Number of splits to create. Usually keep it to the number of machines.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reflection/resources/splits/",
        help="Output directory for the splits.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input directory consisting of all glbs.",
    )
    args = parser.parse_args()
    main(args)