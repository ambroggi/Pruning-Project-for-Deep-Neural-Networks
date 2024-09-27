#!/bin/bash
#SBATCH -c 3  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 23:10:00  # Job time limit
#SBATCH -o results/slurmoutput/slurm-%j.out  # %j = job ID
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH -D /home/abroggi_umassd_edu/Pruning-Project-for-Deep-Neural-Networks
#SBATCH --array=0-8%3  # Run 1 through 8 at a max of 3 at a time


module load miniconda/22.11.1-1
conda activate ModelPrune
python ./main.py --PruningSelection $SLURM_ARRAY_TASK_ID --NumberOfWorkers 3
