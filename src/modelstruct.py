import torch
from . import modelfunctions
import torch.nn.utils.prune


class BaseDetectionModel(torch.nn.Module, modelfunctions.ModelFunctions):
    def __init__(self):
        modelfunctions.ModelFunctions.__init__(self)
        super(BaseDetectionModel, self).__init__()

        self.fc_test_lin = torch.nn.Linear(100, 100)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.fc_test_lin(tensor)


class SimpleCNNModel(BaseDetectionModel):
    def __init__(self):
        modelfunctions.ModelFunctions.__init__(self)
        super(BaseDetectionModel, self).__init__()

        self.conv1 = torch.nn.Conv1d(1, 12, 4)
        self.pool1 = torch.nn.MaxPool1d(3)
        self.pool1.register_parameter("Active_check", param=torch.nn.Parameter(torch.tensor([1], dtype=torch.float, requires_grad=False)))
        self.conv2 = torch.nn.Conv1d(12, 3, 3)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(90, 100)
        self.fc2 = torch.nn.Linear(100, 100)

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
    def __init__(self):
        modelfunctions.ModelFunctions.__init__(self)
        super(BaseDetectionModel, self).__init__()

        self.fc_test_lin = torch.nn.Linear(100, 101)
        self.fc_test_lhidden = torch.nn.Linear(101, 102)
        self.fc_test_lout = torch.nn.Linear(102, 100)

    def forward(self, tensor) -> torch.Tensor:
        x = self.fc_test_lin(tensor)
        y = self.fc_test_lhidden(x)
        z = self.fc_test_lout(y)
        return z


models = {
    "BasicTest": BaseDetectionModel,
    "SwappingTest": SwappingDetectionModel,
    "SimpleCNN": SimpleCNNModel
}


model_layercount = {
    "BasicTest": 1,
    "SwappingTest": 3,
    "SimpleCNN": 4
}


def getModel(name_or_config: str | object, **kwargs) -> BaseDetectionModel:
    if isinstance(name_or_config, str):
        return_val = models[name_or_config](**kwargs)
        layer_count = model_layercount[name_or_config]
    else:
        return_val = models[name_or_config("ModelStructure")](**kwargs)
        return_val.cfg = name_or_config
        layer_count = model_layercount[name_or_config("ModelStructure")]

    # Check that config variables have enough values:
    if len(return_val.cfg("LayerPruneTargets")) != layer_count:
        if len(return_val.cfg("LayerPruneTargets")) > layer_count:
            return_val.cfg("LayerPruneTargets", (return_val.cfg("LayerPruneTargets")[:layer_count]))
        else:
            raise ValueError("config value 'LayerPruneTargets' needs to have at least as many layers as do exist in the model")

    if len(return_val.cfg("WeightPrunePercent")) != layer_count:
        if len(return_val.cfg("WeightPrunePercent")) > layer_count:
            return_val.cfg("WeightPrunePercent", return_val.cfg("WeightPrunePercent")[:layer_count])
        else:
            return_val.cfg("WeightPrunePercent", str(*(return_val.cfg("WeightPrunePercent"), *[1 for _ in range(layer_count - len(return_val.cfg("WeightPrunePercent")))])))

    return return_val


if __name__ == "__main__":
    v = SwappingDetectionModel()
    v.fit()
    v.swap_testlhidden(50)
    v.fit()
