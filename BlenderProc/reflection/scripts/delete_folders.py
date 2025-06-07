import autoroot
import autorootcwd
import os
import time
import shutil
import argparse


def main(args):

    corrupt_uids = set()
    with open(args.file, "r") as f:
        corrupt_uids = set(f.read().splitlines())
    count = 0
    start_time = time.perf_counter()
    # Recursively traverse the input directory
    for root, dirs, files in os.walk(args.dir):
        for d in dirs:
            dir_name = d if not args.multiple_objects else d.split('_')[0]
            if dir_name in corrupt_uids:
                try:
                    shutil.rmtree(os.path.join(root, d))
                    print(f"Deleting {os.path.join(root, d)}")
                    count += 1
                except Exception as e:
                    print(f"Error {e}, deleting {os.path.join(root, d)}")
                    continue

    end_time = time.perf_counter()
    print(f"Number of uid folders deleted: {count}")
    print(f"Time taken: {end_time - start_time} seconds")


def main_resplit(args):
    all_uids = set()
    with open(args.file_all, "r") as f:
        all_uids = set(f.read().splitlines())
    
    small_uids = set()
    with open(args.file_small, "r") as f:
        small_uids = set(f.read().splitlines())

    print(f"Number of all uids: {len(all_uids)}\n"
            f"Number of small uids: {len(small_uids)}")

    print(f"Number of uids in both: {len(all_uids.intersection(small_uids))}")

    # remove small uids from all
    all_uids = all_uids - small_uids

    print(f"Number of uids in all after removing small uids: {len(all_uids)}")

    # write to file
    with open(args.file_all, "w") as f:
        for uid in all_uids:
            f.write(f"{uid}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete uids folders given by list")
    parser.add_argument(
        "--dir",
        type=str,
        default="/data/manan/data/objaverse/blenderproc/hf-objaverse-v1/small_mirrors",
        help="directory to recursively search for uid folders to delete",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="reflection/resources/splits/uids_small_rerun.txt",
        help="file with black uids. Set this to: reflection/resources/splits/uids_objaverse_rerun_0.txt",
    )
    parser.add_argument(
        "--file_all",
        type=str,
        default="reflection/resources/splits/uids_small_rerun.txt",
        help="file with black uids. Set this to: reflection/resources/splits/uids_objaverse_rerun_0.txt",
    )
    parser.add_argument(
        "--file_small",
        type=str,
        default="reflection/resources/splits/uids_small_rerun.txt",
        help="file with black uids. Set this to: reflection/resources/splits/uids_small_rerun.txt",
    )
    parser.add_argument(
        "--multiple_objects",
        action="store_true",
        help="Renders multiple objects",
        default=False,
    )
    args = parser.parse_args()
    main(args)
    # main_resplit(args)