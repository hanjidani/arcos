#!/bin/bash

#SBATCH --job-name=arcos_mnist_wasserstein
#SBATCH --output=exp_8_shifted_mnist_wasserstein/slurm_logs/arcos_mnist_%A_%a.log
#SBATCH --error=exp_8_shifted_mnist_wasserstein/slurm_logs/arcos_mnist_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-59%4

# IMPORTANT: This script must be submitted from the project root directory:
# /home/ali.rasekh/orm/hos/geom/ARCOS
# Example: sbatch exp_8_shifted_mnist_wasserstein/run_on_slurm.sbatch

# Define the absolute path to the conda environment's python
PYTHON_EXEC="/home/ali.rasekh/miniconda3/envs/newenv/envs/arcos/bin/python"

# --- Experiment Grid ---
SHIFT_TYPES=("translation" "rotation")
MODEL_CAPACITIES=(2 4 6 8)
TRANSLATION_SEVERITIES=(0 2 4 6 8 10 12 14)
ROTATION_SEVERITIES=(0 5 10 15 20 25 30)

# --- Create Job Array ---
declare -a PARAMS
for type in "${SHIFT_TYPES[@]}"; do
    if [ "$type" == "translation" ]; then
        for capacity in "${MODEL_CAPACITIES[@]}"; do
            for severity in "${TRANSLATION_SEVERITIES[@]}"; do
                PARAMS+=("$type $capacity $severity")
            done
        done
    elif [ "$type" == "rotation" ]; then
        for capacity in "${MODEL_CAPACITIES[@]}"; do
            for severity in "${ROTATION_SEVERITIES[@]}"; do
                PARAMS+=("$type $capacity $severity")
            done
        done
    fi
done

# --- Slurm Job Array Task ---
NUM_JOBS=${#PARAMS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0} # Default to 0 for single runs

if [ "$TASK_ID" -ge "$NUM_JOBS" ]; then
    echo "Slurm task ID $TASK_ID is out of bounds for array size $NUM_JOBS."
    exit 1
fi

CURRENT_PARAMS=${PARAMS[$TASK_ID]}
set -- $CURRENT_PARAMS
SHIFT_TYPE=$1
MODEL_CAPACITY=$2
SHIFT_SEVERITY=$3

# --- Setup and Execution ---
echo "--- ARCOS MNIST JOB ---"
echo "Job ID: $SLURM_JOB_ID, Task ID: $TASK_ID"
echo "Running on: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Params: $CURRENT_PARAMS"
echo "-----------------------"

# Pre-download the dataset once to prevent race conditions in the job array
echo "Pre-downloading MNIST dataset..."
$PYTHON_EXEC -c "from torchvision import datasets; datasets.MNIST('../data', train=True, download=True); datasets.MNIST('../data', train=False, download=True)"

# Install dependencies
echo "Installing/updating dependencies..."
$PYTHON_EXEC -m pip install -r exp_8_shifted_mnist_wasserstein/requirements.txt

# Create log directory if it doesn't exist
mkdir -p exp_8_shifted_mnist_wasserstein/slurm_logs

# Run the experiment using the full path to python
$PYTHON_EXEC exp_8_shifted_mnist_wasserstein/run_mnist_experiment.py \
    --shift_type $SHIFT_TYPE \
    --model_capacity $MODEL_CAPACITY \
    --shift_severity $SHIFT_SEVERITY \
    --epochs 10

echo "--- Job Finished ---"
