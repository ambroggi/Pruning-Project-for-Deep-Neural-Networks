import torch
import torch.utils.data
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
        test = torch.randint(0, 256, [100], requires_grad=False, dtype=torch.float32), torch.randint(0, 99, [1], requires_grad=False, dtype=torch.long)
        return test

    def __getitems__(self, indexs: list[int]) -> tuple[Targets | torch.Tensor, InputFeatures | torch.Tensor]:
        test = torch.randint(0, 256, [len(indexs), 100], requires_grad=False, dtype=torch.float32), torch.randint(0, 99, [len(indexs)], requires_grad=False, dtype=torch.long)
        return test


def collate_fn_(items):
    if isinstance(items, tuple):
        items = [items]
    features = [i[0] for i in items]
    tags = [i[1] for i in items]
    return (torch.cat(features), torch.cat(tags))


def get_dataset():
    data = VulnerabilityDataset()
    return torch.utils.data.DataLoader(data, batch_size=3, collate_fn=collate_fn_)
