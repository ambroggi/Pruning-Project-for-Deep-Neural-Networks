
# python -m venv ./venv
source .venv/bin/activate
# pip install -r requirements.txt

# Redirecting output if specified
if [[ "$1" == "" ]]
  then
    runname=()
    runnamepath="results/record.csv"
  else
    runname=("--ResultsPath" "$1")
    runnamepath="$1"
fi

# Run the original tests
python ./src/main.py --PruningSelection "0" --NumberOfWorkers 3 --NumberOfEpochs 1 --MaxSamples 0 ${runname[@]} &
python ./src/main.py --PruningSelection "0" --NumberOfWorkers 3 --NumberOfEpochs 1 --MaxSamples 0 ${runname[@]} &
python ./src/main.py --PruningSelection "0" --NumberOfWorkers 3 --NumberOfEpochs 1 --MaxSamples 0 ${runname[@]} &

# Waiting for the originals to finish
wait

# Find the length of the output file so we know what lines the original tests were on
lastline=$(wc -l <$runnamepath)

# Then prune each original model with each possible pruning algorithm
for pruningValue in "ADMM_Joint" "thinet" "iterative_full_theseus" "BERT_theseus" "DAIS" "TOFD" "RandomStructured" "Reduced_Normal_Run"
do
    echo starting $pruningValue tests
    python ./src/main.py --PruningSelection $pruningValue --NumberOfWorkers 3 --NumberOfEpochs 1 --FromSaveLocation "csv $(($lastline-4))" ${runname[@]} >>results/log_a_$pruningValue.txt &
    python ./src/main.py --PruningSelection $pruningValue --NumberOfWorkers 3 --NumberOfEpochs 1 --FromSaveLocation "csv $(($lastline-3))" ${runname[@]} >>results/log_b_$pruningValue.txt &
    python ./src/main.py --PruningSelection $pruningValue --NumberOfWorkers 3 --NumberOfEpochs 1 --FromSaveLocation "csv $(($lastline-2))" ${runname[@]} >>results/log_c_$pruningValue.txt &
    wait
done

echo creating graphs
python ./src/gaphingresults.py $runnamepath

echo making ontologies
python ./src/buildontology.py $runnamepath

echo making ontology graphs
python ./src/ontologygraphing.py