import autoroot
import os
from huggingface_hub import HfApi, HfFileSystem
from pathlib import Path
from loguru import logger as log
import argparse

api = HfApi(token=os.environ["token"])
fs = HfFileSystem(token=os.environ["token"])


def main(args):
    if args.operation == "upload":

        # remote_root = Path(os.path.join("datasets", args.repo_id))
        # all_remote_files = fs.glob(os.path.join("datasets", args.repo_id, "**/*.hdf5"))
        # all_remote_files = [
        #     str(Path(file).relative_to(remote_root)) for file in all_remote_files
        # ]
        # log.info(f"Found {len(all_remote_files)} remote files")
        # args.ignore_patterns.extend(all_remote_files)

        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=args.root_directory,
            private=args.is_private,
            revision=args.revision,
        )

    elif args.operation == "delete":
        api.delete_folder(
            args.path_in_repo,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_description="Delete old folder",
            token=os.environ["token"],
        )


if __name__ == "__main__":
    # python upload2.py
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operation", type=str, default="upload", choices=["upload", "delete"]
    )
    parser.add_argument(
        "--path_in_repo",
        type=str,
        default="hf-objaverse-v1",
        help="Path in the repo to delete",
    )
    parser.add_argument("--repo_id", type=str, default="cs-mshah/SynMirror")
    parser.add_argument(
        "--revision", type=str, default="main", help="Revision to commit to"
    )
    parser.add_argument(
        "--root_directory",
        type=str,
        default="/storage/users/ankitd/data/objaverse/SynMirror",
        help="Root directory to upload (or delete).",
    )
    parser.add_argument(
        "--ignore_patterns",
        help="Patterns to ignore",
        nargs="+",
        default=["spurious", "resources"],
    )
    parser.add_argument(
        "--include_patterns", help="Patterns to include", nargs="+", default=["hdf5"]
    )
    parser.add_argument(
        "--is_private",
        action="store_true",
        help="Whether the dataset is private.",
    )
    args = parser.parse_args()
    main(args)
