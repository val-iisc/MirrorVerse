import os
import numpy as np
import h5py
import argparse
from tqdm import tqdm
import json

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

    # Output Options Group
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output_file",
        type=str, 
        required=True,
        help="Output File where ids which need to be rendered again will be saved",
    )

    return parser


def main(args):
    print("Checking the Renderings now !!!")
    
    uids_with_problematic_renderings = set()
    
    # First pass to collect all HDF5 files to get a total count for tqdm
    all_hdf5_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".hdf5"):
                uid = os.path.basename(root)
                all_hdf5_files.append(os.path.join(root, file))
    


    # Now iterate with tqdm
    print(f"Processing {len(all_hdf5_files)} HDF5 files...")
    for hdf5_path in tqdm(all_hdf5_files, desc="Processing HDF5 files"):
        # Extract root and file from the full path
        root = os.path.dirname(hdf5_path)
        file = os.path.basename(hdf5_path)

        uid = os.path.basename(root) # This assumes UID is the immediate parent directory name
    
        with h5py.File(hdf5_path, "r") as f:
            # Read the data from the file
            structured_array = f['instance_attribute_maps'][()].decode('utf-8') 
            att_list = json.loads(structured_array)

            does_object_exist = False
            for att_map in att_list:
                if att_map["category_id"] > 1: #category map for objects
                    does_object_exist = True
                    break
            
            if not does_object_exist:
                uids_with_problematic_renderings.add(uid)
                        
    
    if len(uids_with_problematic_renderings) > 0:
        print("Rendering Isuses")
        for uid in uids_with_problematic_renderings:
            print(uid)
    else:
        print("No trivial rendering issues.")

    
    if len(uids_with_problematic_renderings) > 0:
        with open(args.output_file, 'w') as f:
            for uid in uids_with_problematic_renderings:
                f.write(f"{uid}\n")
    else:
        print("Nothing to be written in the output file")
        
if __name__ == "__main__":
    parser = get_check_renderings_parser()
    args = parser.parse_args()

    # --- Post-parsing validation and setup ---
    if not os.path.isdir(args.input_dir):
        parser.error(f"Error: Input directory '{args.input_dir}' does not exist or is not a directory.")



    # Example of how to use the parsed arguments
    print(f"Input Directory: {args.input_dir}")
    print(f"Output File: {args.output_file}")


    main(args)