import os
import glob
from PIL import Image
import numpy as np
import h5py
import tqdm
import pandas as pd

# Note : Be careful when using this script. It's for a purpose
#        that paths are hard-code here. 

#Change paths accordingly
MSD_ROOT_DIR = "/data/ankit/Dataset/Reflection/MSD"

IMAGES_DIR = os.path.join(MSD_ROOT_DIR, "images")
MASKS_DIR = os.path.join(MSD_ROOT_DIR, "masks")
CAPTIONS_PATH = os.path.join(MSD_ROOT_DIR, "filtered_captions_edited.csv")
DEPTH_DIR = os.path.join(MSD_ROOT_DIR, "geowizard", "depth_npy")
NORMALS_DIR = os.path.join(MSD_ROOT_DIR, "geowizard", "normal_npy")
HDF5_DIR = os.path.join(MSD_ROOT_DIR, "hdf5")
os.makedirs(HDF5_DIR, exist_ok=True)

#Itrate over images
images_path = glob.glob( os.path.join(IMAGES_DIR, "*.png"))

hdf5_path_list = []
for img_path in tqdm.tqdm(images_path):
    file_name = os.path.basename(img_path).split('.')[0]
    
    hdf5_path = os.path.join(HDF5_DIR,f"{file_name}.hdf5")
    hdf5_path_list.append(hdf5_path)
    if os.path.exists(hdf5_path):
        continue
    mask_path = os.path.join(MASKS_DIR, os.path.split(img_path)[1])
    
    #Goal : Load Image, Mask and save it in hdf5 with dummy depth value
    # data-types = image-uint8 mask-int64 depth-float32
    # mirror id is 1. background is 0
    #keys : 'cam_states', 'category_id_segmaps', 'colors', 'depth', 'instance_attribute_maps'

    image = np.array(Image.open(img_path))
    
    #Loaded mask 255 means it's mirror
    loaded_mask = np.array(Image.open(mask_path))
    mask = np.zeros(loaded_mask.shape, dtype=np.int64)
    mask[loaded_mask>128]=1

    #Load depth
    depth = np.load( os.path.join(DEPTH_DIR, f"{file_name}_pred.npy") )
    normal = np.load( os.path.join(NORMALS_DIR, f"{file_name}_pred.npy") )
    normal = normal.clip(0.,1.)
    # Create an HDF5 file
    cam_state = np.eye(4, dtype=np.float32)
    with h5py.File( hdf5_path, 'w') as h5file:
        # Save the arrays
        h5file.create_dataset('colors', data=image)
        h5file.create_dataset('depth', data=depth)
        h5file.create_dataset('category_id_segmaps', data=mask)
        h5file.create_dataset('normals', data=normal)
        h5file.create_dataset('cam_states', data=cam_state)

print(f"Found {len(hdf5_path_list)} HDF5 files.")

#Relative paths
rel_hdf5_path_list = [os.path.relpath(elem, HDF5_DIR) for elem in hdf5_path_list]
print(rel_hdf5_path_list)
#Create data-frame
test_df = pd.DataFrame(rel_hdf5_path_list, columns=["path"])
test_df["uid"] = test_df["path"].apply(lambda x: x.split('.')[0])
# import captions csv
captions = pd.read_csv(CAPTIONS_PATH)
captions["uid"] = captions["uid"].apply(lambda x: x.split('.')[0])
#captions = captions.rename(columns={'Captions': 'caption'})

# Merge test DataFrame with captions DataFrame on 'uid' column
test_merged_df = pd.merge(test_df, captions[["caption","uid"]], on="uid")

test_save_path = os.path.join(HDF5_DIR, "test_msd.csv")
test_merged_df.to_csv(test_save_path, index=False)