import torch
from torch.utils.data import DataLoader
from numpy import ndarray
import cfg


class ModelFunctions():
    def __init__(self):
        # Non-Overridden values (These actually store things)
        self.epoch_callbacks = []
        self.dataloader = None
        self.optimizer = None

        # Overriden Values (should be overriden by multi-inheritence)
        self.cfg = cfg.ConfigObject()
        self.train = True
        self.parameters = None

    def set_training_data(self, dataloader: DataLoader = None) -> None:
        self.dataloader = dataloader

    def fit(self, epochs: int = 0, dataloader: DataLoader = None) -> None:
        if dataloader is None:
            if self.dataloader is None:
                raise TypeError("No dataset selected for Automatic Vulnerability Detection training")
            dl = self.dataloader
        else:
            dl = dataloader

        # TODO: Configure the optimizer with optimizer specific kwargs
        if self.optimizer is None:
            self.optimizer = self.cfg("Optimizer")(self.parameters(), lr=self.cfg("LearningRate"))

        for batch in dl:
            X, y = batch

    def predict(self, inputs_: torch.Tensor | list[torch.Tensor] | ndarray) -> torch.Tensor | ndarray:
        if not isinstance(inputs_, torch.Tensor):
            with torch.no_grad():
                # This should handle lists of tensors and ndarrays, and then output as an ndarray
                inputs_tensor = torch.tensor(inputs_)
                outputs_tensor = self(inputs_tensor)
                return outputs_tensor.detach().numpy()

        # Check if you want the gradients
        if self.train:
            with torch.no_grad():
                if inputs_.ndim == 1:
                    return self(inputs_.unsqueeze(0))[0]
                else:
                    return self(inputs_)
        else:
            if inputs_.ndim == 1:
                return self(inputs_.unsqueeze(0))[0]
            else:
                return self(inputs_)

    # This should be hidden by Module Inheritence
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        print("Model Functions should be behind Module in the inheritence for the model class.")
        assert False
        return torch.tensor([0])
