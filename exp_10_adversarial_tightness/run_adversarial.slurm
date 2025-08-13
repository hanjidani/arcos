#!/bin/bash
#SBATCH --job-name=adversarial_tightness
# Set a single file for both output and error
#SBATCH --output=exp_10_adversarial_tightness/log/adversarial_tightness_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Explicitly change to the directory where sbatch was submitted
cd $SLURM_SUBMIT_DIR

# Ensure the log directory exists (relative to project root)
mkdir -p exp_10_adversarial_tightness/log

# Manually and robustly set the environment path
# This is a more forceful method than `conda activate` for Slurm
export PATH="/home/ali.rasekh/miniconda3/envs/newenv/envs/arcos/bin:$PATH"

echo "Running adversarial tightness experiment..."
echo "Working directory: $(pwd)"
echo "Using python: $(which python)"

# Run the experiment script using a path relative to the project root
python -u exp_10_adversarial_tightness/run_adversarial_end_to_end.py

echo "Experiment finished."
