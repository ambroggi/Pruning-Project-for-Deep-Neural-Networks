import cfg
import modelstruct
import datareader
import filemanagement


if __name__ == "__main__":
    config = cfg.ConfigObject()
    model = modelstruct.BaseDetectionModel()
    logger = filemanagement.ExperimentLineManager(cfg=config)
    data = datareader.get_dataset()
    model.set_training_data(data)
    model.cfg = config
    # model.epoch_callbacks.append(lambda x: (print(x)))
    model.epoch_callbacks.append(lambda x: ([logger(a, b) for a, b in x.items()]))

    print(model.fit(10))
