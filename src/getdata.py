import os
from functools import partial
import torch
import torch.utils.data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .cfg import ConfigObject


datasets_folder_path = "datasets"


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


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.base = self
        self.scaler = StandardScaler()
        self.number_of_classes = 100
        self.number_of_features = 100
        self.load_kwargs = {"collate_fn": collate_fn_}
        self.scaler_status = 0

    def scale(self, scaler: StandardScaler | None):
        if scaler is not None:
            self.scaler = scaler

        self.dat = pd.DataFrame(self.scaler.transform(self.original_vals), columns=self.original_vals.columns)
        self.dat["label"] = self.original_vals["label"]
        self.scaler_status = 1

    def scale_indicies(self, indicies: list[int]):
        df = self.original_vals.iloc[indicies]
        self.scaler = StandardScaler()
        self.scaler.fit(df)
        self.scale(self.scaler)


class VulnerabilityDataset(BaseDataset):
    def __init__(self, target_format: str = "CrossEntropy"):
        super().__init__()
        self.original_vals = pd.DataFrame([0 for _ in range(30)])
        self.number_of_classes = 100
        self.format = target_format
        pass

    def __len__(self) -> int:
        return len(self.original_vals)

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


class RandomDummyDataset(BaseDataset):
    def __init__(self, target_format: str = "CrossEntropy", num_classes: int = -1):
        super().__init__()
        self.original_vals = pd.DataFrame({"label": [0 for _ in range(3000)]})
        self.number_of_classes = num_classes if num_classes > 0 else 100
        self.format = target_format
        self.rand_seed = torch.randint(0, 100000, [1]).item()
        self.number_of_features = 100
        # self.scale.fit(range(256))
        pass

    def __len__(self) -> int:
        return len(self.original_vals)

    def __getitem__(self, index: int) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        torch.random.manual_seed(self.rand_seed)
        features = torch.randint(0, 256, [self.number_of_features], requires_grad=False, dtype=torch.float32)
        # features.apply_(lambda x: self.scale.transform(x))
        target = torch.randint(0, self.number_of_classes, [1], requires_grad=False, dtype=torch.long)
        if self.format in ["MSE"]:
            target = torch.nn.functional.one_hot(target, num_classes=self.number_of_classes).to(torch.float)
        return features, target

    def __getitems__(self, indexs: list[int]) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        torch.random.manual_seed(self.rand_seed)
        features = torch.randint(0, 256, [len(indexs), self.number_of_features], requires_grad=False, dtype=torch.float32)
        # features.apply_(lambda x: self.scale.transform(x))
        targets = torch.randint(0, self.number_of_classes, [len(indexs)], requires_grad=False, dtype=torch.long)
        if self.format in ["MSE"]:
            targets = torch.nn.functional.one_hot(targets, num_classes=self.number_of_classes).to(torch.float)
        return features, targets


class ACIIOT2023(BaseDataset):
    def __init__(self, target_format: str = "CrossEntropy", num_classes: int = -1, grouped: bool = False, diffrence_multiplier: int | None = None):
        super().__init__()

        if grouped:
            if diffrence_multiplier is None:
                group_str = "-grouped"
                diffrence_multiplier = 10
            else:
                group_str = f"-grouped{diffrence_multiplier}"
        else:
            group_str = ""
            if diffrence_multiplier is None:
                diffrence_multiplier = 100

        if not os.path.exists(os.path.join(datasets_folder_path, f"ACI-IoT-2023{group_str}-formatted-undersampled.parquet")):
            if not os.path.exists(os.path.join(datasets_folder_path, "ACI-IoT-2023-Payload.csv")):
                print("Dataset does not exist please download it from https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023/data?select=ACI-IoT-2023-Payload.csv")
            print("Formatting ACI dataset, this will take some time. eta 25 minutes.")
            self.original_vals = pd.read_csv(os.path.join(datasets_folder_path, "ACI-IoT-2023-Payload.csv"))  # .sample(100000)
            # self.original_vals = pd.read_csv("datasets/ACI-IoT-Example.csv", index_col=0)

            # Looked at (https://www.geeksforgeeks.org/stratified-sampling-in-pandas/) for how to randomly sample from stratified samples
            # Notably I think they were using an older version that .sample() did not work on groups
            # self.original_vals = self.original_vals.groupby("label").sample(n=int(100*min([x for x in self.original_vals.value_counts("label")])), replace=True)
            # It looks like the geeks for geeks article is more of the way to go though, because I need to select a varying number of samples
            if grouped:
                self.original_vals["label"] = self.original_vals["label"].apply(self.grouplabels)
                max_samples = diffrence_multiplier * min(self.original_vals.value_counts("label"))
            else:
                max_samples = diffrence_multiplier * min(self.original_vals.value_counts("label"))
            self.original_vals = self.original_vals.groupby("label", group_keys=False).apply(lambda group: group.sample(n=max_samples if max_samples <= len(group) else len(group)))
            self.original_vals.reset_index(inplace=True)

            # Drop the time column
            self.original_vals.drop(["stime"], inplace=True, axis=1)

            # got how to split the ip columns from: https://stackoverflow.com/a/39358924
            self.original_vals[["src_ip_3", "src_ip_2", "src_ip_1", "src_ip_0"]] = self.original_vals["srcip"].str.split(".", n=3, expand=True).astype(int)
            self.original_vals[["dst_ip_3", "dst_ip_2", "dst_ip_1", "dst_ip_0"]] = self.original_vals["dstip"].str.split(".", n=3, expand=True).astype(int)
            self.original_vals.drop(["srcip", "dstip"], inplace=True, axis=1)

            # revied from (https://www.deeplearningnerds.com/pandas-add-columns-to-a-dataframe-copy/) but I knew .get_dummies() was possible before
            self.original_vals = pd.get_dummies(self.original_vals, columns=["protocol_m"])

            # test = self.vals["payload"].apply(self.from_bytestring)
            # print(test)
            # This whole splitting thing is just because my computer ran out of RAM and using virutal memory slowed this down a lot
            chunk_spliter = [2000*x for x in range(len(self.original_vals)//2000)] + [len(self.original_vals)]
            byte_arrays = pd.concat([self.original_vals["payload"][x1:x2].apply(self.from_bytestring) for x1, x2 in zip(chunk_spliter, chunk_spliter[1:])])
            self.original_vals = pd.concat((self.original_vals, byte_arrays), axis=1)
            self.original_vals.drop(["payload"], inplace=True, axis=1)

            # Picked parquet because of this article: https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d
            # It has best storage size
            self.original_vals.to_parquet(os.path.join(datasets_folder_path, f"ACI-IoT-2023{group_str}-formatted-undersampled.parquet"))
        else:
            self.original_vals = pd.read_parquet(os.path.join(datasets_folder_path, f"ACI-IoT-2023{group_str}-formatted-undersampled.parquet"))

        if not os.path.exists(os.path.join(datasets_folder_path, f"ACI-IoT{group_str}-counts.csv")):
            self.original_vals["label"].value_counts().to_csv(os.path.join(datasets_folder_path, f"ACI-IoT{group_str}-counts.csv"))

        # Get the classes
        self.classes = {label: num for num, label in enumerate(self.original_vals["label"].unique())}
        self.classes.update({num: label for label, num in self.classes.items()})
        self.number_of_classes = len(self.classes)
        self.original_vals["label"] = self.original_vals["label"].map(self.classes)

        self.format = target_format
        # Scalers apparently work well with dataframes? https://stackoverflow.com/a/36475297
        self.scaler.fit(self.original_vals)
        self.dat = pd.DataFrame(self.scaler.transform(self.original_vals), columns=self.original_vals.columns)
        self.dat["label"] = self.original_vals["label"]

        self.number_of_features = len(self.original_vals.columns) - 1

        pass

    def __len__(self) -> int:
        return len(self.original_vals)

    def __getitem__(self, index: int) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        if self.scaler_status == 0:
            print("Warning, no scaler set! please set the scaler!")
            self.scaler_status = -1
        item = self.dat.iloc[index]
        tar = item.pop("label")
        features = torch.Tensor(item.astype(float).to_numpy())
        target = torch.Tensor(tar)
        target = target.long()
        target.requires_grad = False
        if self.format in ["MSE"]:
            target = torch.nn.functional.one_hot(target, num_classes=self.number_of_classes).to(torch.float)
        return features, target

    def __getitems__(self, indexs: list[int]) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        if self.scaler_status == 0:
            print("Warning, no scaler set! please set the scaler!")
            self.scaler_status = -1
        items = self.dat.iloc[indexs]
        tar = items.pop("label")
        features = torch.Tensor(items.astype(float).to_numpy())
        # features.apply_(lambda x: self.scale.transform(x))
        targets = torch.Tensor(tar.astype(int).to_numpy())
        targets = targets.long()
        targets.requires_grad = False
        if self.format in ["MSE"]:
            targets = torch.nn.functional.one_hot(targets, num_classes=self.number_of_classes).to(torch.float)
        return features, targets

    def from_bytestring(self, x):
        # small implementation based on payloadbyte things, https://github.com/Yasir-ali-farrukh/Payload-Byte/blob/main/Pipeline.ipynb
        np_x = np.array(bytearray.fromhex(x), dtype=np.dtype('u1'))
        np_x.resize(1500, refcheck=False)
        series = pd.Series(np_x, index=[f"Byte_{x}" for x in range(1500)], dtype='uint8')
        return series

    @staticmethod
    def grouplabels(old_label):
        if "Flood" in old_label:
            return "Flood"
        if "Scan" in old_label:
            return "Scan"
        else:
            return old_label


def collate_fn_(items: tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor] | list[tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]]) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
    if isinstance(items, tuple):
        items = [items]
    features = [i[0] for i in items]
    tags = [i[1] for i in items]
    return (torch.cat(features), torch.cat(tags))


def get_dataloader(config: ConfigObject | None = None, dataset: None | BaseDataset = None) -> torch.utils.data.DataLoader:
    if dataset is None:
        if config is None:
            print("Config was not given for dataset, creating config")
            config = ConfigObject()
        dataset = get_dataset(config)
    dl = torch.utils.data.DataLoader(dataset, batch_size=config("BatchSize"), num_workers=config("NumberOfWorkers"), persistent_workers=True if config("NumberOfWorkers") > 1 else False, **dataset.base.load_kwargs)
    dl.base = dataset.base
    return dl


def get_dataset(config: ConfigObject) -> BaseDataset:
    datasets: dict[str, torch.utils.data.Dataset] = {"Vulnerability": VulnerabilityDataset, "RandomDummy": RandomDummyDataset, "ACI": ACIIOT2023, "ACI_grouped": partial(ACIIOT2023, grouped=True), "ACI_grouped_fullbalance": partial(ACIIOT2023, grouped=True, diffrence_multiplier=1)}
    data: BaseDataset = datasets[config("DatasetName")](target_format=config("LossFunction", getString=True))
    config("NumClasses", data.number_of_classes)
    config("NumFeatures", data.number_of_features)
    if config("MaxSamples") > 0:
        data, _ = torch.utils.data.random_split(data, [config("MaxSamples"), len(data) - config("MaxSamples")], generator=torch.Generator().manual_seed(1))
        data.base = data.dataset.base
    assert data.base.load_kwargs["collate_fn"] is collate_fn_
    return data


def get_train_test(config: ConfigObject | None = None, dataset: None | BaseDataset = None, dataloader: None | torch.utils.data.DataLoader = None) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset] | tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if config is None:
        print("Config was not given for dataset, creating config")
        config = ConfigObject()
    if dataset is not None and dataloader is not None:
        print("Incorrect usage of get_train_test splitting, please only give a dataset OR a dataloader, not both.")
    elif dataset is not None:
        train, test = torch.utils.data.random_split(dataset, [1 - config("TrainTest"), config("TrainTest")], generator=torch.Generator().manual_seed(1))
        if dataset.base.scaler_status != 1:
            dataset.base.scale_indicies(train.indices)
        train.base = dataset.base
        test.base = dataset.base
        return train, test
    elif dataloader is not None:
        train_ds, test_ds = get_train_test(config=config, dataset=dataloader.dataset)
        return get_dataloader(config=config, dataset=train_ds), get_dataloader(config=config, dataset=test_ds)
    else:
        return get_train_test(config, get_dataset(config))
