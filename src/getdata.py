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


def get_dataloader(config: ConfigObject | None = None, dataset: None | torch.utils.data.Dataset = None) -> torch.utils.data.DataLoader:
    if dataset is None:
        if config is None:
            print("Config was not given for dataset, creating config")
            config = ConfigObject()
        dataset = get_dataset(config)
    return torch.utils.data.DataLoader(dataset, batch_size=config("BatchSize"), num_workers=config("NumberOfWorkers"), **dataset.load_kwargs)


def get_dataset(config: ConfigObject) -> torch.utils.data.Dataset:
    datasets: dict[str, torch.utils.data.Dataset] = {"Vulnerability": VulnerabilityDataset, "RandomDummy": RandomDummyDataset}
    data: torch.utils.data.Dataset = datasets[config("DatasetName")](target_format=config("LossFunction", getString=True))
    if config("MaxSamples") > 0:
        data, _ = torch.utils.data.random_split(data, [config("MaxSamples"), len(data) - config("MaxSamples")], generator=torch.Generator().manual_seed(1))
    data.load_kwargs = {"collate_fn": collate_fn_}
    return data


def get_train_test(config: ConfigObject | None = None, dataset: None | torch.utils.data.Dataset = None, dataloader: None | torch.utils.data.DataLoader = None) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset] | tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if config is None:
        print("Config was not given for dataset, creating config")
        config = ConfigObject()
    if dataset is not None and dataloader is not None:
        print("Incorrect usage of get_train_test splitting, please only give a dataset OR a dataloader, not both.")
    elif dataset is not None:
        train, test = torch.utils.data.random_split(dataset, [config("TrainTest"), 1 - config("TrainTest")], generator=torch.Generator().manual_seed(1))
        train.load_kwargs = {"collate_fn": collate_fn_}
        test.load_kwargs = {"collate_fn": collate_fn_}
        return train, test
    elif dataloader is not None:
        train_ds, test_ds = get_train_test(config=config, dataset=dataloader.dataset)
        return get_dataloader(config=config, dataset=train_ds), get_dataloader(config=config, dataset=test_ds)
    else:
        return get_train_test(config, get_dataset(config))
