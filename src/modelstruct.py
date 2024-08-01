import torch
import modelfunctions
import torch.nn.utils.prune


class BaseDetectionModel(torch.nn.Module, modelfunctions.ModelFunctions):
    def __init__(self):
        modelfunctions.ModelFunctions.__init__(self)
        super(BaseDetectionModel, self).__init__()

        self.fc_test_lin = torch.nn.Linear(100, 100)

    def forward(self, tensor):
        return self.fc_test_lin(tensor)


class SwappingDetectionModel(BaseDetectionModel):
    def __init__(self):
        modelfunctions.ModelFunctions.__init__(self)
        super(BaseDetectionModel, self).__init__()

        self.fc_test_lin = torch.nn.Linear(100, 100)
        self.fc_test_lhidden = torch.nn.Linear(100, 100)
        self.fc_test_lout = torch.nn.Linear(100, 100)

    def forward(self, tensor):
        x = self.fc_test_lin(tensor)
        y = self.fc_test_lhidden(x)
        z = self.fc_test_lout(y)
        return z

    def swap_testlhidden(self, value=50):
        self.fc_test_lhidden = torch.nn.Linear(100, value)

        state_info = self.fc_test_lout.state_dict()
        for name in state_info:
            # print(state_info[name])
            if isinstance(state_info[name], torch.Tensor):
                state_value: torch.Tensor = state_info[name]
                if len(state_value.shape) == 2:
                    state_value = state_value[:, :value]
                # elif len(state_value.shape) == 1:
                #     state_value = state_value[:value]
                state_info[name] = state_value
        self.fc_test_lout = torch.nn.Linear(value, 100)
        self.fc_test_lout.load_state_dict(state_info)
        self.optimizer = None
        print("done")


class SoftPruningLayer():
    def __init__(self, module: torch.nn.Module):
        self.para = torch.nn.Parameter(torch.ones(module.in_features))
        self.module = module
        module.register_parameter(f"v_{module._get_name()}", self.para)
        self.remove_hook = module.register_forward_pre_hook(self)

    def __call__(self, module: torch.nn.Module, args: list[torch.Tensor]):
        return args[0] * self.para[None, :]

    def remove(self):
        self.remove_hook.remove()
        self.module.__getattr__("weight").data *= self.para.data
        self.module.__setattr__(f"v_{self.module._get_name()}", None)
        del self.para


models = {
    "BasicTest": BaseDetectionModel,
    "SwappingTest": SwappingDetectionModel
}


def getModel(name_or_config: str | object, **kwargs) -> BaseDetectionModel:
    if isinstance(name_or_config, str):
        return models[name_or_config](**kwargs)
    else:
        return models[name_or_config("ModelStructure")](**kwargs)


if __name__ == "__main__":
    v = SwappingDetectionModel()
    v.fit()
    v.swap_testlhidden(50)
    v.fit()
