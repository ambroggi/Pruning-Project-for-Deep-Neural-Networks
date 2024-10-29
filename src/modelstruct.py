import torch
import torch.nn.utils.prune

from . import modelfunctions


class BaseDetectionModel(torch.nn.Module, modelfunctions.ModelFunctions):
    """
    The most basic model that also has model functions. This should be overridden.
    """

    def __init__(self, num_classes=100, num_features=100):
        """
        Initialization, inhearits from both torch.nn.Module and from the model functions object which includes all of the functions we want to use.
        """
        modelfunctions.ModelFunctions.__init__(self)
        super(BaseDetectionModel, self).__init__()

        self.fc1 = torch.nn.Linear(num_features, num_classes)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward function should pass the data through all of the modules before returning it as a tensor.
        """
        return self.fc1(tensor)

    def get_important_modules(self) -> torch.nn.ModuleList:
        """
        This is an added function that returns all of the models that should be able to be affected by pruning. This does not count activation functions, Dropout, or Seqentual modules.
        """
        return torch.nn.ModuleList([x for x in self.modules() if isinstance(x, (torch.nn.Linear, torch.nn.Conv1d))])

    def get_important_modules_channelsize(self) -> list[int]:
        """
        Returns a list of integers for how wide each module in get_important_modules() is. This is so that you can set the maximum amount of pruning.
        """
        important = self.get_important_modules()
        sizes = []
        for x in important:
            if isinstance(x, torch.nn.Linear):
                sizes.append(int(x.out_features))
            if isinstance(x, torch.nn.Conv1d):
                sizes.append(int(x.out_channels))

        return sizes

    def load_from_config(self):
        """
        Loads the model from the --FromSaveLocation config value assuming that it was written to. This overrides the current model and thus might cause innacuracies in the output log"""
        if self.cfg("FromSaveLocation") is not None and len(self.cfg("FromSaveLocation")) != 0:
            self.load_model_state_dict_with_structure(torch.load("savedModels/"+self.cfg("FromSaveLocation"), map_location=self.cfg("Device")))


class SimpleCNNModel(BaseDetectionModel):
    """
    A model with a few (2) convolutional layers with maxpooling before two fully connected layers.
    """
    def __init__(self, num_classes=100, num_features=100):
        modelfunctions.ModelFunctions.__init__(self)
        super(SimpleCNNModel, self).__init__()
        del self.fc1

        self.conv1 = torch.nn.Conv1d(1, 12, 4)
        self.pool1 = torch.nn.MaxPool1d(3)
        self.pool1.register_parameter("Active_check", param=torch.nn.Parameter(torch.tensor([1], dtype=torch.float, requires_grad=False)))
        self.conv2 = torch.nn.Conv1d(12, 3, 3)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(int(9*num_features/10), 100)
        self.fc2 = torch.nn.Linear(100, num_classes)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.unsqueeze(dim=1)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class SwappingDetectionModel(BaseDetectionModel):
    """
    Poorly named module that is just three linear layers. Nothing intresintg here as it ws just used for a few simple tests vefore selecting a final model.
    """
    def __init__(self, num_classes=100, num_features=100):
        modelfunctions.ModelFunctions.__init__(self)
        super(BaseDetectionModel, self).__init__()

        self.fc_test_lin = torch.nn.Linear(num_features, 101)
        self.fc_test_lhidden = torch.nn.Linear(101, 102)
        self.fc_test_lout = torch.nn.Linear(102, num_classes)

    def forward(self, tensor) -> torch.Tensor:
        x = self.fc_test_lin(tensor)
        y = self.fc_test_lhidden(x)
        z = self.fc_test_lout(y)
        return z


class LinearLayer(torch.nn.Module):
    """
    A single linear layer with dropout and an activation function for use in the Multilinear layered model
    Otherwise it is an ordinary torch module
    """
    def __init__(self, in_dim, out_dim, dropout):
        torch.nn.Module.__init__(self)
        self.fc = torch.nn.Linear(in_dim, out_dim)
        self.act = torch.nn.LeakyReLU()
        self.drop = torch.nn.Dropout1d(dropout)

    def forward(self, tensor):
        return self.drop(self.act(self.fc(tensor)))


class MultiLinear(BaseDetectionModel):
    """
    Main model used in out tests, variable number of hidden layers with a variable depth.
    """
    def __init__(self, num_classes=100, num_features=100):
        modelfunctions.ModelFunctions.__init__(self)
        super(BaseDetectionModel, self).__init__()

        hdim_size = self.cfg("HiddenDimSize")
        self.fc_lin = torch.nn.Linear(num_features, hdim_size)
        # Add hidden layers
        self.seq = torch.nn.Sequential(*[LinearLayer(hdim_size+x, hdim_size+x+1, self.cfg("Dropout")) for x in range(self.cfg("HiddenDim"))])
        self.fc_lout = torch.nn.Linear(hdim_size+self.cfg("HiddenDim"), num_classes)

    def forward(self, tensor) -> torch.Tensor:
        x = self.fc_lin(tensor)
        y = self.seq(x)
        z = self.fc_lout(y)
        return z


# dictionary of all of the selectable models in the project so far.
models = {
    "BasicTest": BaseDetectionModel,
    "SwappingTest": SwappingDetectionModel,
    "SimpleCNN": SimpleCNNModel,
    "MainLinear": MultiLinear
}


def getModel(name_or_config: str | object, **kwargs) -> BaseDetectionModel:
    """
    Fetches the current model as according to the config (and loads the values if specified with FromSaveLocation) and checks that the config is valid for the model.
    """
    if isinstance(name_or_config, str):
        return_model: BaseDetectionModel = models[name_or_config](**kwargs)
    else:
        if name_or_config("NumClasses") > 0:
            kwargs.update({"num_classes": name_or_config("NumClasses")})
        if name_or_config("NumFeatures") > 0:
            kwargs.update({"num_features": name_or_config("NumFeatures")})
        return_model = models[name_or_config("ModelStructure")](**kwargs)
        return_model.cfg = name_or_config

        if name_or_config("FromSaveLocation") is not None:
            return_model.load_from_config()

    for a, b in zip(return_model.get_important_modules(), [x for x in return_model.modules() if isinstance(x, (torch.nn.Linear, torch.nn.Conv1d))]):
        assert a is b

    validateConfigInModel(return_model)

    return return_model


def validateConfigInModel(model: BaseDetectionModel):
    """
    Validates and sets that the model.cfg config has reasonable values for LayerPruneTargets, LayerIteration, and WeightPrunePercent
    """
    layer_count = len(model.get_important_modules())
    # Check that config variables have enough values:
    for x in ["LayerPruneTargets", "LayerIteration"]:
        config_val = model.cfg(x)
        if len(config_val) != layer_count:
            if len(config_val) > layer_count:
                config_val = config_val[:layer_count]
                model.cfg(x, config_val)
            else:
                config_val.extend([config_val[-1] for _ in range(layer_count - len(config_val))])
                # raise ValueError(f"config value {x} needs to have at least as many layers as do exist in the model")

        # Make sure the channels are bounded by 1 and the original size
        config_val = [int(min(max(target, 1), original)) for target, original in zip(config_val, model.get_important_modules_channelsize())]
        model.cfg(x, config_val)

    for x in ["WeightPrunePercent"]:
        # this sets the amount of the model that remains after pruning
        config_val = model.cfg(x)
        if len(config_val) != layer_count:
            if len(config_val) > layer_count:
                config_val = config_val[:layer_count]
            else:
                # If Nothing is said about the layers assume that none of them will be pruned
                config_val.extend([1 for _ in range(layer_count - len(config_val))])

        # Make sure the percentages are bounded by 0 and 1
        config_val = [float(min(max(target, 0), 1)) for target in config_val]
        model.cfg(x, config_val)

    setlayerprunetargets_to_weightprunepercent(model)


def setlayerprunetargets_to_weightprunepercent(model: BaseDetectionModel):
    """
    Set the ADDM layer prune targets to match the total pruning target.
    """
    # This kind of invalidates that prior "LayerPruneTargets"
    percentages = model.cfg("WeightPrunePercent")
    layer_prune_targets = [int(m*p) for m, p in zip(model.get_important_modules_channelsize(), percentages)]
    model.cfg("LayerPruneTargets", layer_prune_targets)


# These are modules that just be ignored because they are just conatainers for the actual useful modules.
container_modules = (torch.nn.Sequential, LinearLayer)

if __name__ == "__main__":
    v = SwappingDetectionModel()
    v.fit()
    v.swap_testlhidden(50)
    v.fit()
