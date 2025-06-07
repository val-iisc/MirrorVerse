import os
from loguru import logger as log
import argparse


def count_leaf_folders(root_folder):
    count = 0
    try:
        for item in os.listdir(root_folder):
            item_path = os.path.join(root_folder, item)
            if os.path.isdir(item_path):
                if not any(
                    os.path.isdir(os.path.join(item_path, sub_item))
                    for sub_item in os.listdir(item_path)
                ):
                    count += 1
                else:
                    count += count_leaf_folders(item_path)
    except Exception as e:
        log.error(f"An error occurred: {e}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Count leaf folders in a directory.")
    parser.add_argument(
        "--path",
        type=str,
        default="/home/ankitd/data/blenderproc/hf-objaverse-v2",
        help="The root folder path to count leaf folders.",
    )
    args = parser.parse_args()

    root_folder_path = args.path
    total_leaf_folders = count_leaf_folders(root_folder_path)
    log.info(f"Total number of leaf folders: {total_leaf_folders}")


if __name__ == "__main__":
    # python script_name.py --path /home/ankitd/data/blenderproc/hf-objaverse-v2
    main()
