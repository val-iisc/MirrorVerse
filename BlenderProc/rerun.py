import subprocess
import sys
from glob import glob
import os
from typing import List
import json
from loguru import logger as log
import argparse


def parse_args():
    parser = argparse.ArgumentParser("Rerun script for BlenderProc")
    parser.add_argument(
        "--seed",
        help="Seed used for overall generation. Will increment on next subruns to maintain overall randomness.",
    )
    parser.add_argument(
        "params",
        nargs=argparse.REMAINDER,
        help="The params which are handed over to the main.py script",
    )
    args, unknown = parser.parse_known_args()
    return args


def all_complete(cmd_args: List[str]):
    """check if all objects are processed"""
    split_uids = set()
    spurious_uids = set()
    output_dir = cmd_args[cmd_args.index("--output_dir") + 1]
    try:
        split_file = cmd_args[cmd_args.index("--split_file") + 1]
        with open(split_file, "r") as f:
            split_uids = set(f.read().splitlines())

        spurious_file = cmd_args[cmd_args.index("--spurious_file") + 1]
        with open(spurious_file, "r") as f:
            log_data = json.load(f)
            spurious_uids = set(log_data.get("400", {}))
    except Exception as e:
        log.warning(
            f"Error reading split/spurious files: {e}. Please provide split/spurious files."
        )
        # continue running the rerun process
        return False

    # remove spurious from split
    split_uids = split_uids - spurious_uids
    output_objs = glob(f"{output_dir}/*/*/", recursive=True)
    split_file = cmd_args[cmd_args.index("--split_file") + 1]
    if "--multiple_objects" in cmd_args:
        obj_uids_list = [obj.split("/")[-2] for obj in output_objs]
        output_uids = set([obj.split("_")[0] for obj in obj_uids_list])
    else:
        output_uids = set([obj.split("/")[-2] for obj in output_objs])

    # split uids left after removing output uids
    still_to_process = split_uids - output_uids
    log.info(f"Still to process: {len(still_to_process)}")
    return split_uids.issubset(output_uids)


def main():
    # python rerun.py --seed 1234 \
    #     run reflection/main.py \
    #     --camera reflection/resources/cam_novel_poses.txt \
    #     --input_dir ~/data/hf-objaverse-v1/glbs \
    #     --output_dir ~/data/blenderproc/hf-objaverse-v2/ \
    #     --hdri ~/data/blenderproc/resources/HDRI \
    #     --textures ~/data/blenderproc/resources/cc_textures \
    #     --split_file reflection/resources/splits/split_0.txt \
    #     --spurious_file reflection/resources/spurious.json
    args = parse_args()

    seed = None
    env = os.environ
    if args.seed:
        seed = int(args.seed)

    # set the folder in which the cli.py is located
    rerun_folder = os.path.abspath(os.path.dirname(__file__))

    used_arguments = list(args.params)
    # in each run, the arguments are reused
    cmd = ["python", os.path.join(rerun_folder, "cli.py")]
    cmd.extend(used_arguments)
    cmd = " ".join(cmd)

    run = 0
    while not all_complete(used_arguments):
        log.info(f"Subprocess Run {run}: {cmd}")
        if seed:
            log.info(f"Setting seed to {seed}")
            env["BLENDER_PROC_RANDOM_SEED"] = str(seed)
            seed += 1
        # execute one BlenderProc run
        try:
            # keep timeout sufficiently large so that the process is not killed
            subprocess.run(cmd, shell=True, check=True, timeout=3600, env=env)
        except subprocess.CalledProcessError as e:
            log.error(f"Error in run {run}: {e}")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            log.warning("Timeout of 1hr. Restarting process...")
        run += 1


if __name__ == "__main__":
    main()
