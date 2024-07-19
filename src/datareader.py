import torch
import pandas as pd


class Targets(torch.Tensor):
    """
    Should be a 1 dimentisional vector where each value is a int corrisponding to a target class.
    """
    pass


class InputFeatures(torch.Tensor):
    """
    Should be a 2 dimentisional vector where each row is all of the features for a single item.
    """
    pass


class VulnerabilityDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.vals = pd.DataFrame([0 for _ in range(30)])
        pass

    def __len__(self) -> int:
        return len(self.vals)

    def __getitem__(self, index: int) -> tuple[Targets | torch.Tensor, InputFeatures | torch.Tensor]:
        return torch.tensor(0, requires_grad=False), torch.tensor(0, requires_grad=False)

    def __getitems__(self, indexs: list[int]) -> tuple[Targets | torch.Tensor, InputFeatures | torch.Tensor]:
        return torch.tensor([0], requires_grad=False), torch.tensor([0], requires_grad=False)
