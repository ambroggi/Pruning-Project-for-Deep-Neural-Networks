if __name__ != '__mp_main__':
    import src

if __name__ == "__main__":
    kwargs = src.standard_run(save_epoch_waypoints=True)
    # kwargs["model"].save_model_state_dict(logger=kwargs["logger"])
    # load = src.standardLoad()
    # src.standard_run(**load)
    # # swapping_run()
    src.standard_run(PruningSelection="DAIS", **kwargs)
    src.standard_run(PruningSelection="BERT_theseus", **kwargs)
    src.standard_run(PruningSelection="iteritive_full_theseus", **kwargs)
    src.standard_run(PruningSelection="thinet", **kwargs)
    src.standard_run(PruningSelection="ADDM_Joint", **kwargs)
    src.standard_run(**(kwargs | {"logger": None, "NumberOfEpochs": 0}))

    # test = src.standard_run(PruningSelection="DAIS", **kwargs)
    # test = src.standard_run(PruningSelection="iteritive_full_theseus", **test)
    # test = src.standard_run(PruningSelection="thinet", **test)
    # test = src.standard_run(PruningSelection="ADDM_Joint", **test)
    # test = src.standard_run(PruningSelection="BERT_theseus", **test)
    # src.standard_run(**(test | {"logger": None, "NumberOfEpochs": 0}))
