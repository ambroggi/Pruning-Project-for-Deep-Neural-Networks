#!/bin/bash

source .venv/bin/activate
cp results/record.csv results/record_old.csv
rm results/record.csv
python main.py --PruningSelection "0" --LearningRate 0.0009 --NumberOfEpochs 150 --MaxSamples 0
python main.py --PruningSelection "1" --MaxSamples 1000
python main.py --PruningSelection "2" --MaxSamples 1000
python main.py --PruningSelection "3" --MaxSamples 1000
python main.py --PruningSelection "4" --MaxSamples 1000
python main.py --PruningSelection "5" --MaxSamples 1000
python main.py --PruningSelection "6" --MaxSamples 1000
python main.py --PruningSelection "7" --MaxSamples 1000
python main.py --PruningSelection "8" --MaxSamples 1000