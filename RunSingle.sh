#!/bin/bash
#SBATCH -c 3  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 23:10:00  # Job time limit
#SBATCH -o results/slurmoutput/slurm-%j.out  # %j = job ID
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH -D /home/abroggi_umassd_edu/Pruning-Project-for-Deep-Neural-Networks


source .venv/bin/activate
python ./main.py --PruningSelection "0" --NumberOfWorkers 3 --NumberOfEpochs 150 --MaxSamples 0