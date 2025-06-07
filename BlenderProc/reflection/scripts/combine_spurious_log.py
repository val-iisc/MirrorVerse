"""This file combines all spurious log files and creates a single log file"""
import os
import glob
import json
from loguru import logger as log
import argparse

def combine_spurious_log_files(root_folder):
    log_files = glob.glob(os.path.join(root_folder, "spurious_*.json"))
    log_files.sort()
    with open(os.path.join(root_folder, "spurious.json"), "w") as combined_log_file:
        uids_set_400 = set()
        uids_set_300 = set()
        for log_file in log_files:
            with open(log_file, "r") as f:
                log_data = json.load(f)
                uids = list(log_data.get("400", []))
                uids_300 = list(log_data.get("300", []))
                log.info(f"Processing {log_file} with {len(uids) + len(uids_300)} uids")
                # add uids list to uids_set
                uids_set_400.update(uids)
                uids_set_300.update(uids_300)
        log.info(f"Total number of uids in spurious log files: {len(uids_set_400) + len(uids_set_300)}")
        combined_log_file.write(
            json.dumps({"300": list(uids_set_300), "400": list(uids_set_400), "GENERIC_ERROR_CODE": {}}, indent=4)
        )
    log.info("All spurious log files combined into spurious.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine spurious log files")
    parser.add_argument(
        "root_folder",
        type=str,
        default="/data/manan/data/objaverse/blenderproc/hf-objaverse-v1/",
        help="Root folder containing spurious log files",
    )
    args = parser.parse_args()
    combine_spurious_log_files(args.root_folder)