import os
import numpy as np
import h5py
import argparse
from tqdm import tqdm

def get_check_renderings_parser():
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
 

    # Output Options Group
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output_file",
        type=str, 
        required=True,
        help="Output File where ids which need to be rendered again will be saved",
    )

    return parser

def check_number_renderings(uid_dict):
    """
    Checks if all values in a dictionary are equal to 3.

    Args:
        uid_dict (dict): The dictionary to check.

    Returns:
        list: A list of keys whose values are not 3.
    """
    keys_with_non_three_values = []
    for key, value in uid_dict.items():
        if value != 3:
            keys_with_non_three_values.append(key)
    
    if keys_with_non_three_values:
        print("Keys with values not equal to 3:")
        for key in keys_with_non_three_values:
            print(key)
    else:
        print("All UIDS have a value of 3.")
    
    return keys_with_non_three_values

def main(args):
    print("Checking the Renderings now !!!")
    
    #Load the uids from the input file
    count = 0

    if args.input_file is not None:
        if ".txt" in args.input_file:
            with open(args.input_file, 'r') as f:
                extract_uids = f.readlines()
                extract_uids = [f.strip() for f in extract_uids]
                
    print(f"Found {len(extract_uids)} in the input text file.")
    
    
    #Check 1: Count the number of renderings in the folder.
    
    uid_hit = {} #create a dictionary
    
    uids_with_problematic_renderings = set()
    
    # First pass to collect all HDF5 files to get a total count for tqdm
    all_hdf5_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".hdf5"):
                uid = os.path.basename(root)
                if args.input_file is not None and uid not in extract_uids:
                    continue
                
                count += 1
                if uid in uid_hit.keys():
                    uid_hit[uid] += 1
                else:
                    uid_hit[uid] = 1
                all_hdf5_files.append(os.path.join(root, file))
    
    incomplete_render_uids = check_number_renderings(uid_hit)
    
    if len(incomplete_render_uids) > 0:
        return #Fist fix problem with full renderings.
    
    
    # Now iterate with tqdm
    print(f"Processing {len(all_hdf5_files)} HDF5 files...")
    for hdf5_path in tqdm(all_hdf5_files, desc="Processing HDF5 files"):
        # Extract root and file from the full path
        root = os.path.dirname(hdf5_path)
        file = os.path.basename(hdf5_path)

        uid = os.path.basename(root) # This assumes UID is the immediate parent directory name
    
        with h5py.File(hdf5_path, "r") as f:
            # Read the data from the file
            image_data = np.array(f["colors"], dtype=np.uint8)
            category_id_segmaps = np.array(f["category_id_segmaps"], dtype=np.uint8)
            depth_data = np.array(f["depth"])
            normal_data = np.array(f["normals"])
            
            if np.sum(image_data) == 0:
                #All pixels are black
                uids_with_problematic_renderings.add(uid)
                
            if np.sum(category_id_segmaps) == 0:
                #All pixels are black
                uids_with_problematic_renderings.add(uid)
                
            if np.sum(depth_data) == 0:
                #All pixels are black
                uids_with_problematic_renderings.add(uid)
                
            if np.sum(normal_data) == 0:
                #All pixels are black
                uids_with_problematic_renderings.add(uid)
                
            if np.all(normal_data == normal_data.flat[0]):
                uids_with_problematic_renderings.add(uid)
                        
    
    if len(uids_with_problematic_renderings) > 0:
        print("Rendering Isuses")
        for uid in uids_with_problematic_renderings:
            print(uid)
    else:
        print("No trivial rendering issues.")

    uids_with_incomplete_renderings = set(incomplete_render_uids) | uids_with_problematic_renderings
    
    if len(uids_with_incomplete_renderings) > 0:
        with open(args.output_file, 'w') as f:
            for uid in uids_with_incomplete_renderings:
                f.write(f"{uid}\n")
    else:
        print("Nothing to be written in the output file")
        
if __name__ == "__main__":
    parser = get_check_renderings_parser()
    args = parser.parse_args()

    # --- Post-parsing validation and setup ---
    if not os.path.isdir(args.input_dir):
        parser.error(f"Error: Input directory '{args.input_dir}' does not exist or is not a directory.")

    if args.input_file and not os.path.isfile(args.input_file):
        parser.error(f"Error: Input file '{args.input_file}' does not exist or is not a file.")


    # Example of how to use the parsed arguments
    print(f"Input Directory: {args.input_dir}")
    print(f"Input File: {args.input_file}")
    print(f"Output File: {args.output_file}")


    main(args)