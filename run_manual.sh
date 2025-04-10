#!/bin/bash
#SBATCH -c 3  # Number of Cores per Task
#SBATCH --mem=6192  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 1-23:10:00  # Job time limit
#SBATCH -o results/slurmoutput/manual_run-%j-%a.out  # %j = job ID, %a = array id
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH -D /home/abroggi_umassd_edu/Pruning-Project-for-Deep-Neural-Networks
#SBATCH --array=0%3  # Run 1 through 8 at a max of 3 at a time
#SBATCH --nice
#SBATCH --constraint=[2080|2080ti]  # Reccommended for getting the same type of GPUS https://docs.unity.rc.umass.edu/documentation/tools/gpus/ (as of Nov 4 2024)

WEIGHTS=("0.62, 0.56, *, 1")
FILEPATHS=("results/SmallModel(v0.131).csv")
PRUNINGSELECTION=(3)
FILEPATH=${FILEPATHS[$SLURM_ARRAY_TASK_ID]}

# https://www.namehero.com/blog/bash-string-comparison-the-comprehensive-guide/
if [[ "$FILEPATH" == *"Small"* ]]
  then
    SMALLMODEL=("--DatasetName" "ACI_grouped" "--HiddenDimSize" "75")
  else
    SMALLMODEL=()
fi

source .venv/bin/activate
python ./src/main.py --PruningSelection ${PRUNINGSELECTION[$SLURM_ARRAY_TASK_ID]} --NumberOfWorkers 3 --FromSaveLocation "csv $1" --ResultsPath "$FILEPATH" --WeightPrunePercent "${WEIGHTS[$SLURM_ARRAY_TASK_ID]}" --NumberWeightsplits 1 ${SMALLMODEL[@]}