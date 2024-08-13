import src

kwargs = src.standard_run()
kwargs["modelStateDict"] = {a: b.clone() for a, b in kwargs["model"].state_dict().items() if "total" not in a}
# swapping_run()
src.standard_run(PruningSelection="DAIS", **kwargs)
src.standard_run(PruningSelection="BERT_theseus", **kwargs)
src.standard_run(PruningSelection="iteritive_full_theseus", **kwargs)
src.standard_run(PruningSelection="thinet", **kwargs)
src.standard_run(PruningSelection="ADDM_Joint", **kwargs)
src.standard_run(**(kwargs | {"logger": None, "NumberOfEpochs": 0}))
