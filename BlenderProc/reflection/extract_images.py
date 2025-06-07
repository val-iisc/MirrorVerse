import os
import numpy as np
import h5py
from PIL import Image
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../BrushNet/examples/brushnet/dataset"))
from dataset import HDF5Dataset
import pandas as pd
import random

# seed
random.seed(7564)

def main(args):

    count = 0
    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_file is not None:
        if ".txt" in args.input_file:
            # with open(args.input_file, 'r') as f:
            #     extract_uids = f.readlines()
            #     extract_uids = [f.strip() for f in extract_uids]
            extract_uids = [
                    "8343c810f80e42aa849cea818ef1b632_B075X4PTS8_B075X4J118_a0d08e45c4484b46976b44f881f6453d",
                    "429db223039e4464a1bce14d0745be95_19ac02a101dc47968f58aba5eae4dcd2_efcba4fb2d15422580077e2160436d06_4af3c47765af45fd9b0d592a5cb7c7c2",
                    "8343c810f80e42aa849cea818ef1b632_B075X4PTS8",
                    "429db223039e4464a1bce14d0745be95_19ac02a101dc47968f58aba5eae4dcd2_efcba4fb2d15422580077e2160436d06",
                    "4fc697a6dc25426ea920cf89f737a764_a0d08e45c4484b46976b44f881f6453d_5e7c981ad2974772bf85028039ab9d35"
                    ]
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
    parser = argparse.ArgumentParser(description="Extract images from HDF5 files")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data/manan/data/objaverse/blenderproc",
        help="Input directory",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input file containing the uids to extract",
    )
    parser.add_argument("--count", type=int, default=None, help="Number of images to extract")
    parser.add_argument(
        "--output_dir", type=str, default="/data/manan/.cache", help="Output directory"
    )
    parser.add_argument(
        "--extract_mask", action="store_true", help="Extract mask images"
    )
    parser.add_argument(
        "--extract_masked_image", action="store_true", help="Extract masked images"
    )
    parser.add_argument(
        "--extract_depth", action="store_true", help="Extract depth images"
    )
    args = parser.parse_args()

    # args.input_dir = "/home/ankitd/manan/Reflection-Exploration/BrushNet/data/blenderproc"
    args.input_dir = "/home/ankitd/manan/Reflection-Exploration/BrushNet/data/blenderproc/rebuttal"

    # args.input_file = "/home/ankitd/manan/Reflection-Exploration/BrushNet/data/blenderproc/train_objaverse.csv"
    args.input_file = "placeholder.txt"
    # args.count = 200
    args.output_dir = "/home/ankitd/manan/Reflection-Exploration/BrushNet/data/blenderproc/rebuttal_gt_images"

    main(args)