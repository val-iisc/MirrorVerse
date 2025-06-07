import autoroot
import os
from huggingface_hub import HfApi, CommitOperationAdd, HfFileSystem, preupload_lfs_files
from pathlib import Path
from loguru import logger as log
import argparse
import multiprocessing

api = HfApi(token=os.environ["token"])
fs = HfFileSystem(token=os.environ["token"])

def get_all_files(root: Path, include_patterns=[], ignore_patterns=[]):
    def is_ignored(path):
        for pattern in ignore_patterns:
            if pattern in str(path):
                return True
        return False

    def is_included(path):
        for pattern in include_patterns:
            if pattern in str(path):
                return True
        if len(include_patterns) == 0:
            return True
        return False

    dirs = [root]
    while len(dirs) > 0:
        dir = dirs.pop()
        for candidate in dir.iterdir():
            if candidate.is_file() and not is_ignored(candidate) and is_included(candidate):
                yield candidate
            if candidate.is_dir():
                dirs.append(candidate)


def get_groups_of_n(n: int, iterator):
    assert n > 1
    buffer = []
    for elt in iterator:
        if len(buffer) == n:
            yield buffer
            buffer = []
        buffer.append(elt)
    if len(buffer) != 0:
        yield buffer


def main(args):
    if args.operation == "upload":
        # api.create_tag(repo_id=args.repo_id, tag=args.revision, revision="main", exist_ok=True)
        remote_root = Path(os.path.join("datasets", args.repo_id))
        all_remote_files = fs.glob(os.path.join("datasets", args.repo_id, "**/*.hdf5"))
        all_remote_files = [
            str(Path(file).relative_to(remote_root)) for file in all_remote_files
        ]
        log.info(f"Found {len(all_remote_files)} remote files")
        args.ignore_patterns.extend(all_remote_files)

        root = Path(args.root_directory)
        num_threads = args.num_threads
        if num_threads is None:
            num_threads = multiprocessing.cpu_count()
        for i, file_paths in enumerate(get_groups_of_n(args.group_size, get_all_files(root, args.include_patterns, args.ignore_patterns))):
            log.info(f"Committing {len(file_paths)} files...")
            # path_in_repo is path of file_path relative to root_directory
            operations = [] # List of all `CommitOperationAdd` objects that will be generated
            for file_path in file_paths:
                addition = CommitOperationAdd(
                    path_in_repo=str(file_path.relative_to(Path(args.relative_root))),
                    path_or_fileobj=str(file_path),
                )
                preupload_lfs_files(
                    args.repo_id,
                    [addition],
                    token=os.environ["token"],
                    num_threads=num_threads,
                    repo_type="dataset",
                    revision=args.revision,
                )
                operations.append(addition)

            commit_info = api.create_commit(
                repo_id=args.repo_id,
                operations=operations,
                commit_message=f"Upload part {i}",
                repo_type="dataset",
                token=os.environ["token"],
                num_threads=num_threads,
                revision=args.revision,
            )
            log.info(f"Commit {i} done: {commit_info.commit_message}")

    elif args.operation == "delete":
        api.delete_folder(args.path_in_repo, 
                          repo_id=args.repo_id, 
                          repo_type="dataset", 
                          commit_description="Delete old folder", 
                          token=os.environ["token"])

if __name__ == "__main__":
    # python upload.py --root_directory /data/manan/data/objaverse/blenderproc/abo_rendered_data --num_thread 8
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", type=str, default="upload", choices=["upload", "delete"])
    parser.add_argument("--group_size", type=int, default=100)
    parser.add_argument("--repo_id", type=str, default="cs-mshah/SynMirror")
    parser.add_argument(
        "--relative_root",
        type=str,
        default="/data/manan/data/objaverse/blenderproc/",
        help="Relative root",
    )
    parser.add_argument("--revision", type=str, default="main", help="Revision to commit to")
    parser.add_argument("--root_directory", type=str, default="/data/manan/data/objaverse/blenderproc/", help="Root directory to upload (or delete).")
    parser.add_argument("--path_in_repo", type=str, default="hf-objaverse-v1", help="Path in the repo to delete")
    parser.add_argument("--ignore_patterns", help="Patterns to ignore", nargs="+", default=["spurious", "resources"])
    parser.add_argument("--include_patterns", help="Patterns to include", nargs="+", default=["hdf5"])
    parser.add_argument("--num_threads", type=int, default=None, help="Number of threads to use for uploading.")
    args = parser.parse_args()
    main(args)