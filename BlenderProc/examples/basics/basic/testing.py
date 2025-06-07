import h5py
import numpy as np

file = "examples/basics/basic/output/0.hdf5"

with h5py.File(file) as f:
    print(type(f))
    print(f.keys())
    colors = np.array(f["category_id_segmaps"])
    print(colors.shape, colors.dtype, colors.min(), colors.max())