import os
import numpy as np
import h5py
import argparse
from tqdm import tqdm
import json



def main():
    current_working_directory = os.getcwd()

    #Splits
    split_files = [ "resources/splits/large_split_0.txt", 
                    "resources/splits/large_split_1.txt",
                    "resources/splits/large_split_2.txt",
                    "resources/splits/large_split_3.txt",
                    "resources/splits/large_split_4.txt",
                    "resources/splits/large_split_5.txt",
                    "resources/splits/small_split_0.txt",
                    "resources/splits/small_split_1.txt",
                    "resources/splits/small_split_2.txt",
                    "resources/splits/small_split_3.txt",
                    "resources/splits/small_split_4.txt",
                    "resources/splits/small_split_5.txt",
                    "resources/splits/abo/abo_split_0.txt",
                    "resources/splits/abo/abo_split_1.txt"
                  ]
    
    split_maps = {}
    for index, file_name in enumerate(split_files):
        with open( os.path.join(current_working_directory, file_name) , 'r') as f:
            extract_uids = f.readlines()
            extract_uids = [f.strip() for f in extract_uids]
            split_maps[index] = set(extract_uids)
            print(f"Split: {index} \t {file_name} \t Num-Uids: {len(split_maps[index])}")

    print(split_maps.keys())
    count_uids = {}
    for key in split_maps.keys():
        count_uids[key] = set()

    data_dir = "/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/ankit/Reflection/SynMirrorV2"
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".hdf5"):
                uid = os.path.basename(root)
                for key in split_maps.keys():
                    if uid in split_maps[key]:
                        count_uids[key].add(uid)
                        break

    print(count_uids.keys())

    for key in split_maps.keys():
        num_rendered =  len(count_uids[key])
        actual = len(split_maps[key])
        diff = actual - num_rendered
        print(f"Split: {key} Num-Rendered: {num_rendered}\t Actual { actual} \t Difference: { diff}")
        if diff > 0:
            print(split_maps[key] - count_uids[key])
  
        
if __name__ == "__main__":
    main()