#!/bin/bash
#SBATCH --job-name=arcos_cifar_corruption
#SBATCH --output=slurm_logs/arcos_cifar_corruption_%A.out
#SBATCH --error=slurm_logs/arcos_cifar_corruption_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

# Add project root to python path
export PYTHONPATH=$PYTHONPATH:$(dirname $PWD)

# Activate conda environment
source /home/ali.rasekh/miniconda3/etc/profile.d/conda.sh
conda activate newenv

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs

# Install requirements
pip install -r requirements.txt

# Run the experiment
python run_experiment.py --batch_size 256 --num_epochs 10 --lr 0.001
