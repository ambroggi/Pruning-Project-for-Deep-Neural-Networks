import torch
import torch.utils.data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .cfg import ConfigObject


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
    def __init__(self, target_format: str = "CrossEntropy"):
        self.vals = pd.DataFrame([0 for _ in range(30)])
        self.number_of_classes = 100
        self.format = target_format
        pass

    def __len__(self) -> int:
        return len(self.vals)

    def __getitem__(self, index: int) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        features = torch.randint(0, 256, [self.number_of_classes], requires_grad=False, dtype=torch.float32)
        target = torch.randint(0, self.number_of_classes, [1], requires_grad=False, dtype=torch.long)
        if self.format in ["MSE"]:
            target = torch.nn.functional.one_hot(target, num_classes=self.number_of_classes).to(torch.float)
        return features, target

    def __getitems__(self, indexs: list[int]) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        features = torch.randint(0, 256, [len(indexs), self.number_of_classes], requires_grad=False, dtype=torch.float32)
        targets = torch.randint(0, self.number_of_classes, [len(indexs)], requires_grad=False, dtype=torch.long)
        if self.format in ["MSE"]:
            targets = torch.nn.functional.one_hot(targets, num_classes=self.number_of_classes).to(torch.float)
        return features, targets


class RandomDummyDataset(torch.utils.data.Dataset):
    def __init__(self, target_format: str = "CrossEntropy"):
        self.vals = pd.DataFrame([0 for _ in range(3000)])
        self.number_of_classes = 100
        self.format = target_format
        self.rand_seed = torch.randint(0, 100000, [1]).item()
        self.scale = StandardScaler()
        # self.scale.fit(range(256))
        pass

    def __len__(self) -> int:
        return len(self.vals)

    def __getitem__(self, index: int) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        torch.random.manual_seed(self.rand_seed)
        features = torch.randint(0, 256, [self.number_of_classes], requires_grad=False, dtype=torch.float32)
        # features.apply_(lambda x: self.scale.transform(x))
        target = torch.randint(0, self.number_of_classes, [1], requires_grad=False, dtype=torch.long)
        if self.format in ["MSE"]:
            target = torch.nn.functional.one_hot(target, num_classes=self.number_of_classes).to(torch.float)
        return features, target

    def __getitems__(self, indexs: list[int]) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        torch.random.manual_seed(self.rand_seed)
        features = torch.randint(0, 256, [len(indexs), self.number_of_classes], requires_grad=False, dtype=torch.float32)
        # features.apply_(lambda x: self.scale.transform(x))
        targets = torch.randint(0, self.number_of_classes, [len(indexs)], requires_grad=False, dtype=torch.long)
        if self.format in ["MSE"]:
            targets = torch.nn.functional.one_hot(targets, num_classes=self.number_of_classes).to(torch.float)
        return features, targets


def collate_fn_(items: tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor] | list[tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]]) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
    if isinstance(items, tuple):
        items = [items]
    features = [i[0] for i in items]
    tags = [i[1] for i in items]
    return (torch.cat(features), torch.cat(tags))


def get_dataset(config: ConfigObject | None = None) -> torch.utils.data.DataLoader:
    if config is None:
        print("Config was not given for dataset, creating config")
        config = ConfigObject()

    datasets: dict[str, torch.utils.data.Dataset] = {"Vulnerability": VulnerabilityDataset, "RandomDummy": RandomDummyDataset}
    data: torch.utils.data.Dataset = datasets[config("DatasetName")](target_format=config("LossFunction", getString=True))
    data.load_kwargs = {"collate_fn": collate_fn_}
    return torch.utils.data.DataLoader(data, batch_size=config("BatchSize"), num_workers=config("NumberOfWorkers"), **data.load_kwargs)
