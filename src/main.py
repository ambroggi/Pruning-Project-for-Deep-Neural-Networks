if __name__ != '__mp_main__':
    import __init__ as src
    # src.torch.autograd.set_detect_anomaly(True)


def loopAll(config: src.cfg.ConfigObject):
    """
    Runs a standard run and then loops through all of the pruning methods, printing the results to results/record.csv unless otherwise specified by the config.

    Args:
        config (src.cfg.ConfigObject): The config specifications on the runs as a config object
    """
    kwargs = src.standard_run(save_epoch_waypoints=True, config=config)
    kwargs["model"].save_model_state_dict(logger=kwargs["logger"])

    weightsOrdering = range(config("NumberWeightSplits")-1, -1, -1)
    weights = [[                                                    # This builds all of the possible weights percentages
                round(x*((num_variation + 1)/config("NumberWeightSplits"))
                      if x < 1 else 1,                              # Check that the weight should be reduced at all, or if pruning was disabled
                      ndigits=2)                                    # To avoid dealing with large floats, we just round to two decimal places
                if isinstance(x, (float)) else x                    # Sometimes weights are stored as strings, which cannot be rounded
                for x in kwargs["config"]("WeightPrunePercent", getString=True)
                ]
               for num_variation in weightsOrdering
               ]

    for weight_prune_percent in weights:
        kwargs["config"]("WeightPrunePercent", weight_prune_percent)

        src.standard_run(PruningSelection="RandomStructured", **kwargs)
        src.standard_run(PruningSelection="DAIS", **kwargs)
        src.standard_run(PruningSelection="BERT_theseus", **kwargs)
        src.standard_run(PruningSelection="iterative_full_theseus", **kwargs)
        src.standard_run(PruningSelection="thinet", **kwargs)
        src.standard_run(PruningSelection="ADMM_Joint", **kwargs)
        src.standard_run(PruningSelection="TOFD", **kwargs)
        src.standard_run(**(kwargs | {"logger": None, "NumberOfEpochs": 0}))

        # After the first run, assume everything is working properly and no need to check all of the gradients (because it makes it much slower)
        src.torch.autograd.set_detect_anomaly(False)


def loopAlgSpecific(config: src.cfg.ConfigObject, selected: str):
    """
    Runs all of the tests for one specific algorithm based on the selected Prune.
    If selected is "0" or "None" it runs a normal run (no pruning) and saves the model that was made.
    Otherwise it loads according to config("FromSaveLocation") or the first line of the results file if that is none, and prunes the model given there.

    Args:
        config (src.cfg.ConfigObject): The config specifications on the runs as a config object
        selected (str): The algorithm name or index to run.
    """
    config("PruningSelection", "Reset")

    if selected == "None" or selected == "0":
        print("Starting normal run")
        kwargs = src.standard_run(save_epoch_waypoints=True, config=config)
        kwargs["model"].save_model_state_dict(logger=kwargs["logger"])
    else:
        print(f"Starting run of {src.algorithmruns.types_of_tests[selected].__name__}")
        load = src.standardLoad(existing_config=config, index=0)

        weightsOrdering = range(config("NumberWeightSplits")-1, -1, -1)
        weights = [[                                                    # This builds all of the possible weights percentages
                    round(x*((num_variation + 1)/config("NumberWeightSplits"))
                          if x < 1 else 1,                              # Check that the weight should be reduced at all, or if pruning was disabled
                          ndigits=2)                                    # To avoid dealing with large floats, we just round to two decimal places
                    if isinstance(x, (float)) else x                    # Sometimes weights are stored as strings, which cannot be rounded
                    for x in load["config"]("WeightPrunePercent", getString=True)
                    ]
                   for num_variation in weightsOrdering
                   ]

        for weight_prune_percent in weights:
            load["config"]("WeightPrunePercent", weight_prune_percent)
            test = src.standard_run(PruningSelection=selected, **load)
            test

            # After the first run, assume everything is working properly and no need to check all of the gradients (because it makes it much slower)
            src.torch.autograd.set_detect_anomaly(False)


def main():
    config = src.cfg.ConfigObject.get_param_from_args()
    if config("PruningSelection") == "":
        loopAll(config)
    else:
        loopAlgSpecific(config, config("PruningSelection"))


if __name__ == "__main__":
    main()
