import torch
import modelfunctions


class BaseDetectionModel(torch.nn.Module, modelfunctions.ModelFunctions):
    def __init__(self):
        modelfunctions.ModelFunctions.__init__(self)
        super(BaseDetectionModel, self).__init__()

        self.test_lin = torch.nn.Linear(100, 100)

    def forward(self, tensor):
        return self.test_lin(tensor)


if __name__ == "__main__":
    v = BaseDetectionModel()
    v.fit()
