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
    def __init__(self, target_format="CrossEntropy"):
        self.vals = pd.DataFrame([0 for _ in range(30)])
        self.number_of_classes = 100
        self.format = target_format
        pass

    def __len__(self) -> int:
        return len(self.vals)

    def __getitem__(self, index: int) -> tuple[Targets | torch.Tensor, InputFeatures | torch.Tensor]:
        features = torch.randint(0, 256, [self.number_of_classes], requires_grad=False, dtype=torch.float32)
        target = torch.randint(0, self.number_of_classes, [1], requires_grad=False, dtype=torch.long)
        if self.format in ["MSE"]:
            target = torch.nn.functional.one_hot(target, num_classes=self.number_of_classes).to(torch.float)
        return features, target

    def __getitems__(self, indexs: list[int]) -> tuple[Targets | torch.Tensor, InputFeatures | torch.Tensor]:
        features = torch.randint(0, 256, [len(indexs), self.number_of_classes], requires_grad=False, dtype=torch.float32)
        targets = torch.randint(0, self.number_of_classes, [len(indexs)], requires_grad=False, dtype=torch.long)
        if self.format in ["MSE"]:
            targets = torch.nn.functional.one_hot(targets, num_classes=self.number_of_classes).to(torch.float)
        return features, targets


class RandomDummyDataset(torch.utils.data.Dataset):
    def __init__(self, target_format="CrossEntropy"):
        self.vals = pd.DataFrame([0 for _ in range(30)])
        self.number_of_classes = 100
        self.format = target_format
        pass

    def __len__(self) -> int:
        return len(self.vals)

    def __getitem__(self, index: int) -> tuple[Targets | torch.Tensor, InputFeatures | torch.Tensor]:
        features = torch.randint(0, 256, [self.number_of_classes], requires_grad=False, dtype=torch.float32)
        target = torch.randint(0, self.number_of_classes, [1], requires_grad=False, dtype=torch.long)
        if self.format in ["MSE"]:
            target = torch.nn.functional.one_hot(target, num_classes=self.number_of_classes).to(torch.float)
        return features, target

    def __getitems__(self, indexs: list[int]) -> tuple[Targets | torch.Tensor, InputFeatures | torch.Tensor]:
        features = torch.randint(0, 256, [len(indexs), self.number_of_classes], requires_grad=False, dtype=torch.float32)
        targets = torch.randint(0, self.number_of_classes, [len(indexs)], requires_grad=False, dtype=torch.long)
        if self.format in ["MSE"]:
            targets = torch.nn.functional.one_hot(targets, num_classes=self.number_of_classes).to(torch.float)
        return features, targets


def collate_fn_(items):
    if isinstance(items, tuple):
        items = [items]
    features = [i[0] for i in items]
    tags = [i[1] for i in items]
    return (torch.cat(features), torch.cat(tags))


def get_dataset(config=None):
    if config is None:
        print("Config was not given for dataset, importing config")
        import cfg
        config = cfg.ConfigObject()

    datasets = {"Vulnerability": VulnerabilityDataset, "RandomDummy": RandomDummyDataset}
    data = datasets[config("DatasetName")](target_format=config("LossFunction", getString=True))
    return torch.utils.data.DataLoader(data, batch_size=config("BatchSize"), collate_fn=collate_fn_)
