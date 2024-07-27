import torch
import modelfunctions


class BaseDetectionModel(torch.nn.Module, modelfunctions.ModelFunctions):
    def __init__(self):
        modelfunctions.ModelFunctions.__init__(self)
        super(BaseDetectionModel, self).__init__()

        self.test_lin = torch.nn.Linear(100, 100)

    def forward(self, tensor):
        return self.test_lin(tensor)


class SwappingDetectionModel(BaseDetectionModel):
    def __init__(self):
        modelfunctions.ModelFunctions.__init__(self)
        super(BaseDetectionModel, self).__init__()

        self.test_lin = torch.nn.Linear(100, 100)
        self.test_lhidden = torch.nn.Linear(100, 100)
        self.test_lout = torch.nn.Linear(100, 100)

    def forward(self, tensor):
        x = self.test_lin(tensor)
        y = self.test_lhidden(x)
        z = self.test_lout(y)
        return z

    def swap_testlhidden(self, value=50):
        self.test_lhidden = torch.nn.Linear(100, value)

        state_info = self.test_lout.state_dict()
        for name in state_info:
            # print(state_info[name])
            if isinstance(state_info[name], torch.Tensor):
                state_value: torch.Tensor = state_info[name]
                if len(state_value.shape) == 2:
                    state_value = state_value[:, :value]
                # elif len(state_value.shape) == 1:
                #     state_value = state_value[:value]
                state_info[name] = state_value
        self.test_lout = torch.nn.Linear(value, 100)
        self.test_lout.load_state_dict(state_info)
        self.optimizer = None
        print("done")


models = {
    "BasicTest": BaseDetectionModel,
    "SwappingTest": SwappingDetectionModel
}


def getModel(name: str, **kwargs) -> BaseDetectionModel:
    return models[name](**kwargs)


if __name__ == "__main__":
    v = SwappingDetectionModel()
    v.fit()
    v.swap_testlhidden(50)
    v.fit()
