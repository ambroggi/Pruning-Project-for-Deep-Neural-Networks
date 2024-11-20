#!/bin/bash
#SBATCH -c 3  # Number of Cores per Task
#SBATCH --mem=6192  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 1-23:10:00  # Job time limit
#SBATCH -o results/slurmoutput/slurm-%j-%a.out  # %j = job ID, %a = array id
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH -D /home/abroggi_umassd_edu/Pruning-Project-for-Deep-Neural-Networks
#SBATCH --array=1-8%3  # Run 1 through 8 at a max of 3 at a time
#SBATCH --nice
#SBATCH --constraint=[2080|2080ti]  # Reccommended for getting the same type of GPUS https://docs.unity.rc.umass.edu/documentation/tools/gpus/ (as of Nov 4 2024)

source .venv/bin/activate
python ./main.py --PruningSelection $SLURM_ARRAY_TASK_ID --NumberOfWorkers 3 --FromSaveLocation "csv $1" --ResultsPath "$2"
