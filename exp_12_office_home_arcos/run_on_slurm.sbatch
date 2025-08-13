#!/bin/bash
#SBATCH --job-name=arcos_office_home
#SBATCH --output=slurm_logs/arcos_office_home_%A.out
#SBATCH --error=slurm_logs/arcos_office_home_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1

# Add project root to python path
export PYTHONPATH=$PYTHONPATH:$(dirname $PWD)

# Activate conda environment
source /home/ali.rasekh/miniconda3/etc/profile.d/conda.sh
conda activate newenv

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs

# Run the experiment
python run_experiment.py --batch_size 64 --num_epochs 15 --lr 0.001 --bad_lr 0.1 --weight_decay 1e-4
