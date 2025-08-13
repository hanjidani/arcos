#!/bin/bash
#SBATCH --job-name=exp9_feature_shift
#SBATCH --output=log/exp9_feature_shift_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-8

# Activate the conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate arcos
echo "Conda environment activated."

# Define the severities to test
SEVERITIES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Get the severity for the current job array task
CURRENT_SEVERITY=${SEVERITIES[$SLURM_ARRAY_TASK_ID]}

echo "Starting experiment for shift_severity: $CURRENT_SEVERITY"

# Run the main experiment script
python run_feature_shift_experiment.py \
    --shift_severity $CURRENT_SEVERITY \
    --epochs 50

echo "Experiment finished for shift_severity: $CURRENT_SEVERITY"
