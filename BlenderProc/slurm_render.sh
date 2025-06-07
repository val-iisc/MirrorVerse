#!/bin/bash
#SBATCH --job-name=disable_rotate_split_4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=11:59:00
#SBATCH --partition=long
#SBATCH --gres=gpu:2
#SBATCH -w cn3

module load conda/24.1.2 cuda/cuda12.4

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
# Activate the conda environment
conda activate brushnet
pip install loguru

# Run GPU monitoring every 30 seconds in the background
# Set log file with job name and job ID
GPU_LOG_FILE="logs/gpu_usage_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,nounits -l 30 > $GPU_LOG_FILE &

python rerun.py \
    run reflection/main.py \
    --camera reflection/resources/cam_novel_poses.txt \
    --input_dir ~/data/hf-objaverse-v1/glbs \
    --output_dir ~/data/blenderproc/hf-objaverse-v4/ \
    --hdri ~/data/blenderproc/resources/HDRI \
    --textures ~/data/blenderproc/resources/cc_textures \
    --split_file reflection/resources/splits/disable_rotate_split_4.txt \
    --spurious_file reflection/resources/spurious.json \
    --max_render_time 45 \
    --disable_rotate

# Kill GPU monitoring after the job is done
kill %1

echo "Slurm Job ID: $SLURM_JOB_ID Job Name: $SLURM_JOB_NAME finished"