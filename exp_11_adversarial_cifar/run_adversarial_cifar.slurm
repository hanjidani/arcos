#!/bin/bash
#SBATCH --job-name=adversarial_cifar
# Set a single file for both output and error
#SBATCH --output=exp_11_adversarial_cifar/log/adversarial_cifar_%j.log
#SBATCH --error=exp_11_adversarial_cifar/log/adversarial_cifar_%j.log
#SBATCH --time=02:00:00 # Increased time for larger dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Explicitly change to the directory where sbatch was submitted
cd $SLURM_SUBMIT_DIR

# Ensure the log directory exists
mkdir -p exp_11_adversarial_cifar/log

# Manually and robustly set the environment path
export PATH="/home/ali.rasekh/miniconda3/envs/newenv/envs/arcos/bin:$PATH"

echo "Running Adversarial CIFAR-10 experiment..."
echo "Working directory: $(pwd)"
echo "Using python: $(which python)"

# Run the experiment script
python -u exp_11_adversarial_cifar/run_adversarial_cifar.py

echo "Experiment finished."
