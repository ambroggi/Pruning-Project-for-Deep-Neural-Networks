import torch
import modelfunctions


class VolnerabilityModel(torch.nn.Module, modelfunctions.ModelFunctions):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        return tensor


if __name__ == "__main__":
    v = VolnerabilityModel()
    v.fit()
