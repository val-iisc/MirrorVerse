# 🧪 Dataset Curation using Blender & BlenderProc


This directory contains scripts for generating synthetic reflection datasets using BlenderProc. Make sure your working directory is `Reflection-Exploration/BlenderProc`.

> 🔗 For an overview of BlenderProc, refer to the `examples/` folder in the [BlenderProc repository](https://github.com/DLR-RM/BlenderProc).

## 📚 Table of Contents

- [🧪 Dataset Curation using Blender & BlenderProc](#-dataset-curation-using-blender--blenderproc)
  - [🔧 Environment Setup](#-environment-setup)
  - [✨ Dataset Sources and Aesthetic filtering](#-dataset-sources-and-aesthetic-filtering)
  - [🏗️ Data Generation](#️-data-generation)
    - [Basic Command](#basic-command)
    - [Recommended (for robust and reproducible runs)](#recommended-for-robust-and-reproducible-runs)
    - [🔧 Command Line Arguments](#-command-line-arguments)
    - [Boolean Flags](#boolean-flags)
    - [Visualize Outputs](#visualize-outputs)
  - [📊 Dataset Construction & Upload](#-dataset-construction--upload)
    - [Upload to HuggingFace](#upload-to-huggingface)
    - [Faster Uploads](#faster-uploads)
  - [🖼️ Extract Images & Visualize](#️-extract-images--visualize)
    - [Convert HDF5 to PNG](#convert-hdf5-to-png)
    - [Visualize Using FiftyOne](#visualize-using-fiftyone)
  - [📂 Public Dataset](#-public-dataset)
  
---

## 🔧 Environment Setup

```bash
conda create -n blender python=3.10
conda activate blender
pip install -e .
blenderproc pip install debugpy hydra-core==1.3.2 hydra-colorlog==1.2.0 loguru
pip install fiftyone loguru simple-aesthetics-predictor tqdm autoroot autorootcwd
```

## ✨ Dataset Sources and Aesthetic filtering
We provide utilities to filter objects based on aesthetic scores.

Scripts:

 - scripts/download_renderings.py: to fetch renderings.

- scripts/predict_aesthetics.py: to score renderings.

💡 You can optionally download the renderings manually from this [link](https://tri-ml-public.s3.amazonaws.com/datasets/views_release.tar.gz).

💡 You can download the Amazon Berkley Objects from this [link](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)


## 🏗️ Data Generation

During execution, **Blender** will be installed if not already present (required by BlenderProc).

### Basic Command
```bash
blenderproc run reflection/main.py
```

### Recommended (for robust and reproducible runs)
```bash
python rerun.py --seed 1234 \
    run reflection/main.py \
    --camera reflection/resources/cam_novel_poses.txt \
    --input_dir ~/data/hf-objaverse-v1/glbs \
    --output_dir ~/data/blenderproc/hf-objaverse-v2/ \
    --hdri ~/data/blenderproc/resources/HDRI \
    --textures ~/data/blenderproc/resources/cc_textures \
    --split_file reflection/resources/splits/split_0.txt \
    --spurious_file reflection/resources/spurious.json
```

#### 🔧 Command Line Arguments

This script accepts several command-line arguments for controlling rendering, scene setup, and dataset configuration:

##### Required / Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--camera` | Path to the camera pose file used for rendering. | `reflection/resources/cam_poses.txt` |
| `--mirror` | Path to the mirror `.glb` file (scene geometry). | `reflection/resources/all_mirrors.glb` |
| `--hdri` | Directory containing HDRI environment maps. | `/data/manan/data/objaverse/blenderproc/resources/HDRI` |
| `--textures` | Directory containing floor textures. | `blenderproc/resources/textures` |
| `--object` | Path to a specific 3D object file (usually `.glb`). | `reflection/resources/objaverse_examples/063b1b7d877a402ead76cedb06341681/063b1b7d877a402ead76cedb06341681.glb` |
| `--input_dir` | Directory containing all input 3D objects. | `reflection/resources/objaverse_examples/` |
| `--split_file` | Optional path to a dataset split file. | `""` |
| `--num_render` | Number of renderings per object. | `3` |
| `--spurious_file` | JSON file listing spurious objects to exclude. | `/data/manan/data/objaverse/blenderproc/hf-objaverse-v1/spurious_0.json` |
| `--output_dir` | Output directory for saving generated data. | `reflection/output/blenderproc` |
| `--max_objects` | Max number of objects to process in this run. | `75` |
| `--max_time` | Maximum allowed processing time (in minutes). | `30` |
| `--max_render_time` | Timeout for rendering a single object (in seconds). | `30` |
| `--model_3d_type` | File format of the 3D model (`glb`, `obj`, `fbx`). | `glb` |
| `--seed` | Random seed for reproducibility. | `None` |

### Boolean Flags

These flags are optional and toggle specific behaviors when provided.

| Flag | Description |
|------|-------------|
| `--small_mirrors` | Randomly select mirrors from a small-mirrors subset. |
| `--disable_rotate` | Prevent automatic rotation of objects. |
| `--fast_testing` | Enable quick rendering with reduced quality (for debugging). |
| `--single_run` | Perform only a single rendering operation for test/debug. |
| `--reprocess` | Reprocess an object even if already present in the output. **(⚠ Avoid using with `rerun.py`)** |
| `--check_spurious` | Dynamically check for spurious objects during import. |
| `--multiple_objects` | Render scenes with multiple objects instead of just one. |
| `--create_rotate_trans_test_set` | Create a test set specifically for evaluating rotation and translation understanding. |

---

### Visualize Outputs
```bash
blenderproc vis hdf5 reflection/output/blenderproc/0.hdf5
```


## 📊 Dataset Construction & Upload

Final splits are created via `dataset.ipynb` using [cuDF-pandas](https://rapids.ai/cudf-pandas/).  
👉 [Install RAPIDS](https://docs.rapids.ai/install) before usage.

### Upload to HuggingFace

Create a `.env` file containing your HuggingFace token:

```env
HF_TOKEN=<your_token>
```

Then run:

```bash
python reflection/upload.py
```

### Faster Uploads

```bash
pip install 'huggingface_hub[hf_transfer]'
```

Update `.env`:

```env
HF_HUB_ENABLE_HF_TRANSFER=1
HF_HUB_ETAG_TIMEOUT=500
```

---

## 🖼️ Extract Images & Visualize

### Convert HDF5 to PNG
The `extract_images.py` script can be used to convert the generated HDF5 files into PNG images. This script offers several command-line arguments to control the input, output, and types of images extracted.

Command Line Arguments for extract_images.py

| Argument | Type | Description |
|---|---|---|
| `--input_dir` | `str` | Input directory containing HDF5 files to extract images from. |
| `--input_file` | `str` | Optional path to a file containing UIDs (e.g., a list of object IDs) to selectively extract images for. If `None`, all found HDF5 files in `input_dir` will be processed. |
| `--count` | `int` | Number of images to extract. If `None`, all available images will be extracted. |
| `--output_dir` | `str` | Output directory where the extracted PNG images will be saved. |
| `--extract_mask` | `store_true` | Flag to extract **mask images** (binary segmentation masks). |
| `--extract_masked_image` | `store_true` | Flag to extract **masked images** (e.g., foreground objects with a black background). |
| `--extract_depth` | `store_true` | Flag to extract **depth images** (grayscale images representing distance from the camera). |

#### Usage Examples:

**1. Extract all image types from a specific input directory to a specified output directory:**

```bash
python reflection/extract_images.py \
    --input_dir <input_dir_path> \
    --output_dir <output_dir_path> \
    --extract_mask --extract_masked_image --extract_depth
```
### Visualize Using [FiftyOne](https://docs.voxel51.com)

#### On Remote

```bash
pip install fiftyone
python reflection/visualise.py
```

#### On Local

```bash
pip install "fiftyone[desktop]"
fiftyone app connect --destination test@<remote-ip>
```

> You can tag images (e.g., `flag`) and revisit them later via filter UI.

---

## 📂 Public Dataset

- Dataset hosted on HuggingFace:  
  👉 [Mirror-Fusion/Objaverse-Mirrors](https://huggingface.co/datasets/Mirror-Fusion/Objaverse-Mirrors)
