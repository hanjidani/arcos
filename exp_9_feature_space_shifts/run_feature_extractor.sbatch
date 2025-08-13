#!/bin/bash
#SBATCH --job-name=train_feature_extractor
#SBATCH --output=log/train_feature_extractor_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

# Activate the conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate arcos
echo "Conda environment activated."

# Run the feature extractor training script
echo "Starting feature extractor training..."
python -u feature_extractor.py
echo "Feature extractor training finished."
