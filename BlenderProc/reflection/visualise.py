import fiftyone as fo
import os
import argparse
from glob import glob

def main(args):

    if args.load_saved:
        dataset = fo.load_dataset("Mirror-Dataset")
    else:
        file_paths = glob(os.path.join(args.data_dir, "**", "*.png"), recursive=True)

        dataset = fo.Dataset.from_images(
            list(file_paths),
            name="Mirror-Dataset",
            overwrite=True,
            persistent=True,
            progress=True,
        )

    session = fo.launch_app(dataset, remote=True, port=args.port)
    session.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from HDF5 files")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/manan/.cache/hf-objaverse-v1",
        help="Data directory",
    )
    parser.add_argument(
        "--load_saved",
        action="store_true",
        help="After creating the fo.Dataset once, \
          use this flag to load the already created dataset",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5151,
        help="Port number for the FiftyOne session",
    )
    args = parser.parse_args()
    main(args)