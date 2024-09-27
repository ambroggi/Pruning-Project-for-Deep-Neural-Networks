#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 00:10:00  # Job time limit
#SBATCH -o results/slurmoutput/slurm-%j.out  # %j = job ID
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH -D /home/abroggi_umassd_edu/Pruning-Project-for-Deep-Neural-Networks


module load miniconda/22.11.1-1
conda activate ModelPrune
python ./main.py
