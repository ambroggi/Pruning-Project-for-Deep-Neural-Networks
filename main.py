import src

kwargs = src.standard_run()
kwargs["modelStateDict"] = {a: b.clone() for a, b in kwargs["model"].state_dict().items() if "total" not in a}
# swapping_run()
src.standard_run(PruningSelection="Theseus", **kwargs)
src.standard_run(PruningSelection="thinet", **kwargs)
src.standard_run(PruningSelection="ADDM_Joint", **kwargs)
