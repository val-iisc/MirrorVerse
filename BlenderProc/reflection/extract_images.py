import os
import numpy as np
import h5py
from PIL import Image
import argparse
import sys
from scene_compositor.dataset import HDF5Dataset
import pandas as pd
import random

# seed
random.seed(7564)


def get_image_extraction_parser():
    parser = argparse.ArgumentParser(description="Extract images from HDF5 files")

    # Input Options Group
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--input_dir",
        type=str, 
        required=True,
        help="Input directory containing HDF5 files.",
    )
    input_group.add_argument(
        "--input_file",
        type=str,
        default=None, 
        help="Input file containing the UIDs to extract (e.g., a text file with one UID per line).",
    )
    input_group.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of images to extract. If not specified, all images in input_file (or all found) will be extracted.",
    )

    # Output Options Group
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output_dir",
        type=str, 
        required=True,
        help="Output directory where extracted images will be saved.",
    )

    # Extraction Type Options Group
    extraction_group = parser.add_argument_group("Extraction Type Options")
    extraction_group.add_argument(
        "--extract_mask",
        action="store_true",
        help="Extract mask images (binary segmentation masks).",
    )
    extraction_group.add_argument(
        "--extract_masked_image",
        action="store_true",
        help="Extract images with masks applied (e.g., foreground objects only).",
    )
    extraction_group.add_argument(
        "--extract_depth",
        action="store_true",
        help="Extract depth images (grayscale images representing distance).",
    )

    return parser

def main(args):

    count = 0

    if args.input_file is not None:
        if ".txt" in args.input_file:
            with open(args.input_file, 'r') as f:
                extract_uids = f.readlines()
                extract_uids = [f.strip() for f in extract_uids]
        elif ".csv" in args.input_file:
            df = pd.read_csv(args.input_file)
            keywords = ['chair', 'sofa', 'cuboid', 'box', 'chest', 'table', 'cabinet', 'desk', 'stool', 'cupboard']
            # create a subset of df which has any of the keywords in the string of the 'auto_caption' column
            df = df[df['auto_caption'].str.contains('|'.join(keywords))]

            extract_uids = df['uid'].tolist()
            print(f'df with keywords shape: {df.shape}')
            # shuffle the list
            random.shuffle(extract_uids)

    # Recursively traverse the input directory
    for root, dirs, files in os.walk(args.input_dir):
        if args.count is not None and count >= args.count:
            print(f'Extracted {count} images')
            break
        for file in files:
            if file.endswith(".hdf5"):
                uid = os.path.basename(root)
                if args.input_file is not None and uid not in extract_uids:
                    continue
                # Construct the full path to the HDF5 file
                hdf5_path = os.path.join(root, file)

                # Create the corresponding output directory
                output_subdir = os.path.join(
                    args.output_dir,
                    os.path.relpath(root, args.input_dir),
                )
                os.makedirs(output_subdir, exist_ok=True)

                # Construct the output image path
                filename = file.split('.')[0]
                rgb_path = os.path.join(
                    output_subdir, f"{filename}.png"
                )
                if os.path.exists(rgb_path):
                    continue

                # Open the HDF5 file
                with h5py.File(hdf5_path, "r") as f:
                    print(f"Extracting RGB images to {rgb_path}")
                    # Read the image data from the file
                    image_data = np.array(f["colors"], dtype=np.uint8)
                    # Save the image to the output path
                    Image.fromarray(image_data).save(rgb_path)

                    if args.extract_mask:
                        mask_path = os.path.join(
                            output_subdir, f"{filename}_mask.png"
                        )
                        mask_data = (
                            np.array(f["category_id_segmaps"], dtype=np.uint8) == 1
                        ).astype(np.uint8) * 255
                        Image.fromarray(mask_data).save(mask_path)

                    if args.extract_masked_image:
                        masked_image_path = os.path.join(
                            output_subdir, f"{filename}_masked.png"
                        )
                        masked_image = image_data.copy()
                        masked_image[mask_data == 255] = 0
                        Image.fromarray(masked_image).save(masked_image_path)

                    if args.extract_depth:
                        depth_path = os.path.join(
                            output_subdir, f"{filename}_depth.npy"
                        )
                        depth_data = np.array(f["depth"])
                        np.save(depth_path, depth_data)
                    if args.count is not None:
                        count += 1
                        if count >= args.count:
                            break


if __name__ == "__main__":
    parser = get_image_extraction_parser()
    args = parser.parse_args()

    # --- Post-parsing validation and setup ---
    if not os.path.isdir(args.input_dir):
        parser.error(f"Error: Input directory '{args.input_dir}' does not exist or is not a directory.")

    if args.input_file and not os.path.isfile(args.input_file):
        parser.error(f"Error: Input file '{args.input_file}' does not exist or is not a file.")

    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        parser.error(f"Error: Could not create output directory '{args.output_dir}': {e}")

    # Example of how to use the parsed arguments
    print(f"Input Directory: {args.input_dir}")
    print(f"Input File: {args.input_file}")
    print(f"Count: {args.count}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Extract Mask: {args.extract_mask}")
    print(f"Extract Masked Image: {args.extract_masked_image}")
    print(f"Extract Depth: {args.extract_depth}")


    main(args)