#!/bin/bash
# https://stackoverflow.com/a/6482403
if [ $# -eq 0 ]
  then
    path_to_use="None"
  else
    path_to_use="$1"
fi

mainjoba=$(sbatch --parsable --nice RunSingle.sh "$path_to_use")
mainjobb=$(sbatch --parsable --nice RunSingle.sh "$path_to_use")
mainjobc=$(sbatch --parsable --nice RunSingle.sh "$path_to_use")

sbatch --dependency=aftercorr:$mainjoba:$mainjobb:$mainjobc --nice RunBatchOfSingle.sh 0 "$path_to_use"
sbatch --dependency=aftercorr:$mainjoba:$mainjobb:$mainjobc --nice RunBatchOfSingle.sh 1 "$path_to_use"
sbatch --dependency=aftercorr:$mainjoba:$mainjobb:$mainjobc --nice RunBatchOfSingle.sh 2 "$path_to_use"