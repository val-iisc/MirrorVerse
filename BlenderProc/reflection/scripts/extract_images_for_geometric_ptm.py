import os
import numpy as np
import h5py
from PIL import Image
import argparse
import pandas as pd
from tqdm import tqdm


def get_hdf5_list(csv_file_path):
    # Step 1: Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Access the first column directly by indexing
    first_column_data = df.iloc[:, 0]  # iloc[:, 0] selects all rows of the first column

    return first_column_data.tolist()

def main(args):
    #Read csv file
    hdf5_list = get_hdf5_list(args.input_csv)
    
    os.makedirs(args.output_dir, exist_ok=True)

    masked_path = os.path.join(args.output_dir, "masked_image")
    full_img_path = os.path.join(args.output_dir, "full_image")
    os.makedirs(masked_path, exist_ok=True)
    os.makedirs(full_img_path, exist_ok=True)

    def create_unique_filename(token_name):
        #abo_v3/B/B07HSLGGDB/0.hdf5
        uid, f_name = os.path.split(token_name)
        uid = os.path.split(uid)[1]
        return f"{uid}_{f_name.split('.')[0]}.png"

    for hdf5_file in tqdm(hdf5_list):
        hdf5_path = os.path.join(args.input_dir,  hdf5_file)
        if not os.path.exists(hdf5_path):
            print(f"Path does not exist : {hdf5_path}")
            continue

        #Open the HDF5 file
        with h5py.File(hdf5_path, "r") as f:
            # Read the image data from the file
            image_data = np.array(f["colors"], dtype=np.uint8)

            f_name = f"{create_unique_filename(hdf5_file)}"
            full_image_output_path = os.path.join(full_img_path, f_name)
            # Save the image to the output path
            Image.fromarray(image_data).save(full_image_output_path)

            #Create mask image
            mask = np.array(f["category_id_segmaps"], dtype=np.uint8)
            masked_image_data = image_data.copy()
            masked_image_data[mask == 1] = 0
            masked_image_output_path = os.path.join(masked_path, f_name)
            Image.fromarray(masked_image_data).save(masked_image_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from HDF5 files")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="csv file for extracting images ",
    )
    parser.add_argument(
        "--output_dir", type=str, 
        required=True, help="Output directory"
    )
    parser.add_argument("--masked_imgs", action="store_true", default=False, help="Save masked images")
    
    args = parser.parse_args()
    main(args)