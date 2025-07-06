import os
import numpy as np
import h5py
from PIL import Image
import argparse
import pandas as pd
import random
import matplotlib.pyplot as plt

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
        help="Extract depth images.",
    )
    extraction_group.add_argument(
        "--extract_normal",
        action="store_true",
        help="Extract normal map images.",
    )

    return parser

def save_depth_map(data_array, output_path, cmap='turbo', vmin=None, vmax=None):
    """
    Saves a 2D NumPy array as a PNG image, applying a specified colormap.

    Args:
        data_array (np.ndarray): The 2D NumPy array to save.
        output_path (str): The full path including filename and .png extension.
        cmap (str or matplotlib.colors.Colormap): The colormap to apply. Defaults to 'viridis'.
        vmin (float, optional): The minimum value for colormap normalization.
        vmax (float, optional): The maximum value for colormap normalization.
    """
    if not isinstance(data_array, np.ndarray) or data_array.ndim != 2:
        raise ValueError("Input data_array must be a 2D NumPy array.")

    # Use plt.imsave to directly save the array as an image
    # It handles colormapping and normalization internally
    plt.imsave(output_path, data_array, cmap=cmap, vmin=vmin, vmax=vmax)
    
def save_normal_map(normal_map_array, output_path):
    """
    Saves a NumPy array representing a normal map as an RGB PNG image.
    Normal maps are typically float arrays with values from -1 to 1 per channel.
    This function remaps them to 0-1 for display.

    Args:
        normal_map_array (np.ndarray): A 3D NumPy array of shape (height, width, 3)
                                       representing the normal map. Values should be
                                       in the range [-1, 1].
        output_path (str): The full path including filename and extension (e.g., 'output/normal_map.png').
    """
    if not isinstance(normal_map_array, np.ndarray) or normal_map_array.ndim != 3 or normal_map_array.shape[2] != 3:
        raise ValueError("Input normal_map_array must be a 3D NumPy array of shape (height, width, 3).")

    # Remap values from [-1, 1] to [0, 1] for image saving
    # The common mapping for normal maps: (N + 1) / 2
    # Where N is the normal component (-1 to 1)
    display_normal_map = (normal_map_array + 1) / 2.0
    
    # Ensure values are strictly within [0, 1] after remapping,
    # due to potential floating point inaccuracies or slight out-of-range values.
    display_normal_map = np.clip(display_normal_map, 0, 1)

    # plt.imsave directly saves the RGB array. No colormap needed.
    plt.imsave(output_path, display_normal_map)
    
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
                            output_subdir, f"{filename}_depth.png"
                        )
                        depth_data = np.array(f["depth"])
                        save_depth_map(depth_data, depth_path, cmap='turbo', vmin=0, vmax=6)
                    if args.extract_normal:
                        normal_path = os.path.join(
                            output_subdir, f"{filename}_normal.png"
                        )
                        normal_data = np.array(f["normals"])
                        save_normal_map(normal_data, normal_path)
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