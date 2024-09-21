if __name__ != '__mp_main__':
    import src

if __name__ == "__main__":
    config = src.cfg.ConfigObject.get_param_from_args()
    if config("PruningSelection") == "":
        kwargs = src.standard_run(save_epoch_waypoints=True, config=config)
        kwargs["model"].save_model_state_dict(logger=kwargs["logger"])

        for weight_prune_percent in [[round(x*((num_variation + 1)/8) if x < 1 else 1, ndigits=2) if isinstance(x, (float)) else x for x in kwargs["config"]("WeightPrunePercent", getString=True)] for num_variation in range(7, -1, -1)]:
            kwargs["config"]("WeightPrunePercent", weight_prune_percent)

            src.standard_run(PruningSelection="RandomStructured", **kwargs)
            src.standard_run(PruningSelection="DAIS", **kwargs)
            src.standard_run(PruningSelection="BERT_theseus", **kwargs)
            src.standard_run(PruningSelection="iteritive_full_theseus", **kwargs)
            src.standard_run(PruningSelection="thinet", **kwargs)
            src.standard_run(PruningSelection="ADDM_Joint", **kwargs)
            src.standard_run(PruningSelection="TOFD", **kwargs)
            src.standard_run(**(kwargs | {"logger": None, "NumberOfEpochs": 0}))

    else:
        selected = config("PruningSelection")
        config("PruningSelection", "Reset")

        if selected == "None" or selected == "0":
            kwargs = src.standard_run(save_epoch_waypoints=True, config=config)
            kwargs["model"].save_model_state_dict(logger=kwargs["logger"])
        else:
            load = src.standardLoad(existing_config=config, index=0)
            for weight_prune_percent in [[round(x*((num_variation + 1)/8) if x < 1 else 1, ndigits=2) if isinstance(x, (float)) else x for x in load["config"]("WeightPrunePercent", getString=True)] for num_variation in range(7, -1, -1)]:
                load["config"]("WeightPrunePercent", weight_prune_percent)
                test = src.standard_run(PruningSelection=selected, **load)
