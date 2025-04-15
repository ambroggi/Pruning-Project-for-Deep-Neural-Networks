# Structure of Repository

This repository is structured into several folders and several run-able files.

Folders:
- datasets: The dataset files (`https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023/data`) must be placed here in order to allow the model to run.
    - datasets/ontologies: A folder that caches some of the ontologies built from existing models using `src/buildontology.py`
- results: The logging files that store all run information.
    - 500-Random: results from running `src/buildontology.py` on 500 randomly connected ontologies as a baseline. 
    - results/images: This folder stores all of the graphs we tried to generate, but only after you run `src/graphingresults.py` that generates the images.
    - results/raytraining: Stores info about raytrain runs, raytrain is a library that ties to find good hyperparameters that we used. See `src/raytraining.py`
    - results/waypointing: Some results from running the ontology search on waypoints, snapshots of the model during training, to see changes over training time.
- savedModels: The main models that were used in runs will be stored here with a unique name. Exact models and historical performance can be inferred from the resulting logging file.
- src: Stores all of the code.
    - src/Imported_Code: Stores all code for running the specific algorithms, each file is labeled with if it is original or from another source. Although most are at least slightly modified.

Files:
- .git*: Files that identify specific git information.
- *.sh: Files for running the main python files on the High Performance Computing cluster that was available to us.
- `main.py`: This is the main file and will be what you need to run to launch our code and generate data from pruning. Use `python main.py -h` to learn specific command line options.
- `buildontology.py` and `ontologygraphing.py`: These two files are how we analyzed the model with ontological methods.
- README.md: This file. You are reading it already, thank you.
- requirements.txt: Lists modules that you need to install with pip. Use `python -m venv .venv; source .venv/bin/activate; pip install -r requirements.txt` to install everything in case you don't know how.


# Pruning Algorithms applied to Network Intrusion Detection Dataset

This is a pipeline for testing various different pruning methods on an arbitrary Artificial Neural Network (ANN) model for the purposes of reducing the required runtime and memory usage. This will hopefully make smaller ANN models for Network Intrusion Detection Systems (NIDS) that can fit in mobile platforms.

## Algorithms used

ADMM [link](https://github.com/leegs52/admm_joint_pruning) (*Code Used and Slightly Modified*)

BERT_Theseus [link](https://arxiv.org/pdf/2002.02925)

DAIS [link](https://arxiv.org/pdf/2011.02166)

Iterative Theseus (*Created brute force technique*)

ThiNet [link](https://github.com/Roll920/ThiNet_Code/tree/master) (*Code Used, Adapting code also made but kept separate*)

TOFD [link](https://github.com/ArchipLab-LinfengZhang/Task-Oriented-Feature-Distillation/commit/fcfd4be5ff773d2d27adccdc7df206cdf502800e) *Code Used and Slightly Modified*

Random Pruning *Standard Method*

Complete replacement *Baseline training new model*


## How to run

There are several meanings to the phrase "running" here as there are three python files that are set up to be run directly and several shell files that will run batches of tests.

### Dependencies

In order to run anything you need to install the required libraries and submodules.
Note: Only the libraries, step 3, are required to graph results, the rest are required to generate your own results.

1. Install git submodules. Use `git submodule init; git submodule update` to install submodules. These are extra code that could be copied directly from the original code source so that there were minimal modifications. Due to that they are not directly included in the download and must be installed separately.

2. Download the dataset. Download the dataset from `https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023/data?select=ACI-IoT-2023-Payload.csv` and place the payload csv file into the datasets/ folder.

3. Install requirements.txt. Use `python -m venv .venv; source .venv/bin/activate; pip install -r requirements.txt` to install all required dependencies into a virtual environment. Note that your command line should now say (.venv) at the start. You will need to rerun `source .venv/bin/activate` any time that it does not have (.venv) when using this repository as the commands given install all of the requirements into a virtual environment that will need to be re-started each time. 

4. Initial run. The dataset needs to be formatted correctly. This means that the first run of the code will be significantly slower than the rest before it is re-saved in a more densely organized format. we recommend only having one instance running during this time period to avoid any kind of write-corruption. Future runs can be run concurrently. 

### Graphing our results

The results we got from our original runs can be re-graphed by running the `src/graphingresults.py` file. This will generate all of the graphs in the images folder as well as the LaTeX table, saved as results/images/table.txt, that we used in our paper.

### Generating your own data

You can easily generate your own data by using `python main.py`. This has several options for each run. You can use `python main.py -h` to see the full options list with explanations but the main ones are:

- PruningSelection: If PruningSelection is set, then only that one algorithm will be run. If nothing is set then the code will go round robbin over all of the algorithms (slow for parallelization). Algorithm names are as follows (see bottom of `src/algorithmruns.py` file):

    0. Train the original model.
    1. ADMM
    2. TOFD
    3. IterativeTheseus
    4. BERT_Theseus
    5. DAIS
    6. ThiNet
    7. Random Pruning
    8. Complete replacement
- ResultsPath: This specifies what file the results will be printed into, or read from. The default is `results/record.csv`.
- FromSaveLocation: This specifies what model to load. You can either have a specific model name from savedModels, or `csv x` where x is the row index of the model from the ResultsPath csv to look for the final model of. Only training the original model mode will save a final model by default so you may need to change the code if you want to stack changes on top of each other. Also note that if you load directly from a specific model, not a csv, it will not load the specific model properties as those are stored in the csv, not the model file.

### SLURM

We had access to a High Performance Computing (HPC) cluster, which made our tests run faster. If you also have access to a HPC that runs SLURM for batching jobs, you may be able to just run our batch job shell files, with modifications to the running directory as that is an absolute path. Note that using these batch files is not necessary to run our code, they are available if you want to and are able to use them. We recommend just using `fullrun.sh` but you may need to run specific tests if the time limits run out.

### Hyperparameter Tuning

Tuning hyperparameters was done by using the raytraining module. This was set up and run using the file `src/raytraining.py`. You can run this file and look at the output to find the best case for Learning Rate, Dropout rate, Hidden layer count, and Hidden layer size. This was more of a finicky process so we are unable to write down the whole steps for finding the best hyperparameters from this output.


# Ontology Observation of Pruned Models

This is a method of observing our pruned and unpruned models using rdflib ontologies. It consists of the two files `src/buildontology.py` and `src/ontologygraphing.py`