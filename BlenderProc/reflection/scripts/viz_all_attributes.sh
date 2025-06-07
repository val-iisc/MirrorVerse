#!/bin/bash

# Directory to search for .hdf5 files
DIRECTORY=$1
OUTPUT_FOLDER=$2


# Check if the directory is provided
if [ -z "$DIRECTORY" ]; then
  echo "Usage: $0 <directory> <output_folder>"
  exit 1
fi

# Check if the directory is provided
if [ -z "$OUTPUT_FOLDER" ]; then
  echo "Usage: $0 <directory> <output_folder>"
  exit 1
fi

mkdir -p $OUTPUT_FOLDER
# Initialize an array to store .hdf5 file paths
hdf5_files=()

# Find .hdf5 files and store them in the array
while IFS= read -r -d '' file; do
  hdf5_files+=("$file")
done < <(find "$DIRECTORY" -type f -name "*.hdf5" -print0)

# Process all files in the array
for file in "${hdf5_files[@]}"; do
  filename=$(basename "$file")
  parent_folder=$(dirname "$file")
  relative_path=$(realpath --relative-to="$DIRECTORY" "$parent_folder")
  echo "File: $filename, Parent Folder: $parent_folder, Relative Path : $relative_path"
  blenderproc vis hdf5 $file --save $OUTPUT_FOLDER/$relative_path/$filename
done