import torch
import pandas as pd


class VulnerabilityDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.vals = pd.DataFrame()
        pass

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return 0, 0

    def __getitems__(self, indexs: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        return [0], [0]
