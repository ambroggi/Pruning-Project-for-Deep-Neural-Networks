import os
from functools import partial
from itertools import groupby

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.preprocessing import StandardScaler

try:
    from .cfg import ConfigObject
except ImportError as e:
    if "no known parent package" in e.args[0]:
        from cfg import ConfigObject
    else:
        raise

datasets_folder_path = "datasets"


class Targets(torch.Tensor):
    """
    Should be a 1 dimensional vector where each value is a int corresponding to a target class.
    Like this: [1, 4, 2, 1, 1, 0]
    """
    pass


class InputFeatures(torch.Tensor):
    """
    Should be a 2 dimensional vector where each row is all of the features for a single item.
    Like this: [
        [0.3, 0.6, 0.1],
        [0.9, 1, 0.2],
        [0.9, 0.1, 0.4]
    ]
    """
    pass


class ModifiedDataloader(torch.utils.data.DataLoader):
    base: "BaseDataset"


class BaseDataset(torch.utils.data.Dataset):
    """This is the base abstract dataset type that is for all datasets for this code base.
    """
    def __init__(self):
        self.base = self  # Find the original dataset even after splitting (this is to apply the scaler and find the original after splits)
        self.scaler = StandardScaler()
        self.number_of_classes = 100
        self.number_of_features = 100
        self.load_kwargs = {"collate_fn": collate_fn_}
        self.scaler_status = 0

        self.classes = {}
        self.feature_labels = {}
        self.format = None
        self.original_vals: pd.DataFrame

    def scale(self, scaler: StandardScaler | None = None):
        """Scale the dataset into a normalized form. This new dataset is saved as self.dat

        Args:
            scaler (StandardScaler | None): A scaler to apply to the data. If None it uses the default StandardScaler()
        """
        if scaler is not None:
            self.scaler = scaler

        self.dat = pd.DataFrame(self.scaler.transform(self.original_vals), columns=self.original_vals.columns)
        self.dat["label"] = self.original_vals["label"]
        self.scaler_status = 1

    def scale_indices(self, indices: list[int]):
        """Create a scaler from specific indices of the data and save that scaler to self.scaler. Then applies that scaler.

        Args:
            indices (list[int]): list of indices to use for the fitting of the scaler.
        """
        df = self.original_vals.iloc[indices]
        self.scaler = StandardScaler()
        self.scaler.fit(df)
        self.scale(self.scaler)

    def target_to_one_hot(self, targets: torch.Tensor):
        if self.format in ["MSE"]:
            targets = torch.nn.functional.one_hot(targets, num_classes=self.number_of_classes).to(torch.float)
        return targets


class SingleClass(BaseDataset):
    def __init__(self, items: list[torch.Tensor], class_idx: int, original_dataset: BaseDataset):
        self.base = self  # reset the base, because scaling should have already been applied.
        self.number_of_features = original_dataset.number_of_features
        self.load_kwargs = original_dataset.load_kwargs
        self.feature_labels = original_dataset.feature_labels

        self.classes = original_dataset.classes
        self.number_of_classes = len(self.classes)
        self.dat = items
        self.target = torch.tensor(class_idx)
        self.format = original_dataset.format

    def __len__(self) -> int:
        return len(self.dat)

    def __getitem__(self, index: int) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        features = self.dat[index]
        target = self.target_to_one_hot(self.target)
        return features, target

    def __getitems__(self, indexes: list[int]) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        features = torch.stack([self.dat[x] for x in indexes])
        targets = self.target_to_one_hot(self.target.expand(len(indexes)))
        return features, targets


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
        target = self.target_to_one_hot(target)
        return features, target

    def __getitems__(self, indexes: list[int]) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        features = torch.randint(0, 256, [len(indexes), self.number_of_classes], requires_grad=False, dtype=torch.float32)
        targets = torch.randint(0, self.number_of_classes, [len(indexes)], requires_grad=False, dtype=torch.long)
        targets = self.target_to_one_hot(targets)
        return features, targets


class RandomDummyDataset(BaseDataset):
    """This dataset is just full of random numbers to make sure everything is working properly"""
    def __init__(self, target_format: str = "CrossEntropy", num_classes: int = -1):
        """Initialize the random dummy dataset.

        Args:
            target_format (str, optional): Type of output format for the targets to use. This changes depending on the loss function being used. Defaults to "CrossEntropy".
            num_classes (int, optional): The number of classes to randomly generate. If <1 it picks 100. Defaults to -1.
        """
        super().__init__()
        self.original_vals = pd.DataFrame({"label": [0 for _ in range(3000)]})
        self.number_of_classes = num_classes if num_classes > 0 else 100
        self.format = target_format
        self.rand_seed = torch.randint(0, 100000, [1]).item()
        self.number_of_features = 100
        # self.scale.fit(range(256))
        pass

    def __len__(self) -> int:
        """Gets the length of the dataset

        Returns:
            int: Number of items in the dataset
        """
        return len(self.original_vals)

    def __getitem__(self, index: int) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        """Gets a single item from the dataset.

        Args:
            index (int): Value to get specifically from the dataset

        Returns:
            tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]: a tuple containing input features and a target value (target is either a tensor or a vector depending on how the dataloader was set up)
        """
        torch.random.manual_seed(self.rand_seed)
        features = torch.randint(0, 256, [self.number_of_features], requires_grad=False, dtype=torch.float32)
        # features.apply_(lambda x: self.scale.transform(x))
        target = torch.randint(0, self.number_of_classes, [1], requires_grad=False, dtype=torch.long)
        target = self.target_to_one_hot(target)
        return features, target

    def __getitems__(self, indexes: list[int]) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        """Gets a list of items from the dataset using a list of indices

        Args:
            indexes (list[int]): List of integers  pointing at indices from the dataset, must be available in the data so < len(self)

        Returns:
            tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]: Tuple containing a tensor of features and a tensor of targets
        """
        torch.random.manual_seed(self.rand_seed)
        features = torch.randint(0, 256, [len(indexes), self.number_of_features], requires_grad=False, dtype=torch.float32)
        # features.apply_(lambda x: self.scale.transform(x))
        targets = torch.randint(0, self.number_of_classes, [len(indexes)], requires_grad=False, dtype=torch.long)
        targets = self.target_to_one_hot(targets)
        return features, targets


class ACIIOT2023(BaseDataset):
    """This dataset is the Army Cyber Institute - Internet Of Things 2023 data"""

    def __init__(self, target_format: str = "CrossEntropy", num_classes: int = -1, grouped: bool = False, difference_multiplier: int | None = None):
        """Initialize the ACI dataset, this includes checking if class grouping (combining related smaller classes into one) is working. a
        As well as formatting the data if it has not already been saved in the formatted version, which includes:
            - Undersampling if needed to balance the dataset
            - Dropping the time column
            - Splitting payloads and IP addresses up by byte
            - Turning protocol into a one-hot vector
            - And saving the formatted version as a parquet file
        Then some final formatting steps are performed:
            - Relabeling targets as integers instead of strings
            - Identifying target target format and matching it
            - Scaling the data

        Args:
            target_format (str, optional): Type of output format for the targets to use. This changes depending on the loss function being used. Defaults to "CrossEntropy".
            num_classes (int, optional): The number of classes to randomly generate. If <1 it picks 100. Defaults to -1.
        """
        super().__init__()

        if grouped:
            if difference_multiplier is None:
                group_str = "-grouped"
                difference_multiplier = 10
            else:
                group_str = f"-grouped{difference_multiplier}"
        else:
            group_str = ""
            if difference_multiplier is None:
                difference_multiplier = 100

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
                self.original_vals["label"] = self.original_vals["label"].apply(self.group_labels)
                max_samples = difference_multiplier * min(self.original_vals.value_counts("label"))
            else:
                max_samples = difference_multiplier * min(self.original_vals.value_counts("label"))
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
            # This whole splitting thing is just because my computer ran out of RAM and using virtual memory slowed this down a lot
            chunk_splitter = [2000*x for x in range(len(self.original_vals)//2000)] + [len(self.original_vals)]
            byte_arrays = pd.concat([self.original_vals["payload"][x1:x2].apply(self.from_bytestring) for x1, x2 in zip(chunk_splitter, chunk_splitter[1:])])
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
        self.number_of_classes = len(self.classes)
        self.classes.update({num: label for label, num in self.classes.items()})
        self.original_vals["label"] = self.original_vals["label"].map(self.classes)

        self.format = target_format
        # Scalers apparently work well with dataframes? https://stackoverflow.com/a/36475297
        self.scaler.fit(self.original_vals)
        self.dat = pd.DataFrame(self.scaler.transform(self.original_vals), columns=self.original_vals.columns)
        self.dat["label"] = self.original_vals["label"]

        self.number_of_features = len(self.original_vals.columns) - 2  # -1 extra for index
        self.feature_labels = {feature_num: column_name for feature_num, column_name in enumerate(filter(lambda x: x not in {"label", "index"}, self.original_vals.columns))}

        pass

    def __len__(self) -> int:
        return len(self.original_vals)

    def __getitem__(self, index: int) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        if self.scaler_status == 0:
            print("Warning, no scaler set! please set the scaler!")
            self.scaler_status = -1
        item = self.dat.iloc[index]
        item.pop("index")
        tar = item.pop("label")
        features = torch.Tensor(item.astype(float).to_numpy())
        target = torch.Tensor(tar)
        target = target.long()
        target.requires_grad = False
        target = self.target_to_one_hot(target)
        return features, target

    def __getitems__(self, indexes: list[int]) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        if self.scaler_status == 0:
            print("Warning, no scaler set! please set the scaler!")
            self.scaler_status = -1
        items = self.dat.iloc[indexes]
        items.pop("index")
        tar = items.pop("label")
        features = torch.Tensor(items.astype(float).to_numpy())
        # features.apply_(lambda x: self.scale.transform(x))
        targets = torch.Tensor(tar.astype(int).to_numpy())
        targets = targets.long()
        targets.requires_grad = False
        targets = self.target_to_one_hot(targets)
        return features, targets

    def from_bytestring(self, x):
        # small implementation based on payloadByte things, https://github.com/Yasir-ali-farrukh/Payload-Byte/blob/main/Pipeline.ipynb
        np_x = np.array(bytearray.fromhex(x), dtype=np.dtype('u1'))
        np_x.resize(1500, refcheck=False)
        series = pd.Series(np_x, index=[f"Byte_{x}" for x in range(1500)], dtype='uint8')
        return series

    @staticmethod
    def group_labels(old_label):
        if "Flood" in old_label:
            return "Flood"
        if "Scan" in old_label:
            return "Scan"
        else:
            return old_label


class ACIPayloadless(BaseDataset):
    """This dataset is the Army Cyber Institute - Internet Of Things 2023 data but does not have the payload data"""

    def __init__(self, target_format: str = "CrossEntropy", num_classes: int = -1, grouped: bool = False, difference_multiplier: int | None = None):
        """Initialize the ACI dataset, this includes checking if class grouping (combining related smaller classes into one) is working. a
        As well as formatting the data if it has not already been saved in the formatted version, which includes:
            - Undersampling if needed to balance the dataset
            - Dropping the time column
            - Splitting payloads and IP addresses up by byte
            - Turning protocol into a one-hot vector
            - And saving the formatted version as a parquet file
        Then some final formatting steps are performed:
            - Relabeling targets as integers instead of strings
            - Identifying target target format and matching it
            - Scaling the data

        Args:
            target_format (str, optional): Type of output format for the targets to use. This changes depending on the loss function being used. Defaults to "CrossEntropy".
            num_classes (int, optional): The number of classes to randomly generate. If <1 it picks 100. Defaults to -1.
        """
        super().__init__()

        if grouped:
            if difference_multiplier is None:
                group_str = "-grouped"
                difference_multiplier = 10
            else:
                group_str = f"-grouped{difference_multiplier}"
        else:
            group_str = ""
            if difference_multiplier is None:
                difference_multiplier = 100

        if not os.path.exists(os.path.join(datasets_folder_path, f"ACI-IoT-2023-flow{group_str}-formatted-undersampled.parquet")):
            if not os.path.exists(os.path.join(datasets_folder_path, "ACI-IoT-2023.xlsx")):
                print("Dataset does not exist please download it from https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023/data")
            print("Formatting ACI dataset, this will take some time. eta 25 minutes.")
            self.original_vals = pd.read_excel(os.path.join(datasets_folder_path, "ACI-IoT-2023.xlsx"), sheet_name="ACI-IoT-2023")  # .sample(100000)
            # self.original_vals = pd.read_csv("datasets/ACI-IoT-Example.csv", index_col=0)

            # Looked at (https://www.geeksforgeeks.org/stratified-sampling-in-pandas/) for how to randomly sample from stratified samples
            # Notably I think they were using an older version that .sample() did not work on groups
            # self.original_vals = self.original_vals.groupby("label").sample(n=int(100*min([x for x in self.original_vals.value_counts("label")])), replace=True)
            # It looks like the geeks for geeks article is more of the way to go though, because I need to select a varying number of samples
            if grouped:
                self.original_vals["Label"] = self.original_vals["Label"].apply(self.group_labels)
                max_samples = difference_multiplier * min(self.original_vals.value_counts("Label"))
            else:
                max_samples = difference_multiplier * min(self.original_vals.value_counts("Label"))
            self.original_vals = self.original_vals.groupby("Label", group_keys=False).apply(lambda group: group.sample(n=max_samples if max_samples <= len(group) else len(group)))
            self.original_vals.reset_index(inplace=True)

            # Drop the ID value, time, and connection type columns
            self.original_vals.drop(["Flow ID", "Timestamp", "Connection Type"], inplace=True, axis=1)

            # Set infinity "Flow Bytes/s" and "Flow Packets/s"
            self.original_vals["Flow Bytes/s"] = self.original_vals["Flow Bytes/s"].map(lambda x: -1 if np.isinf(x) or np.isnan(x) else x)
            self.original_vals["Flow Packets/s"] = self.original_vals["Flow Packets/s"].map(lambda x: -1 if np.isinf(x) or np.isnan(x) else x)

            # got how to split the ip columns from: https://stackoverflow.com/a/39358924
            self.original_vals[["src_ip_3", "src_ip_2", "src_ip_1", "src_ip_0"]] = self.original_vals["Src IP"].str.split(".", n=3, expand=True).astype(int)
            self.original_vals[["dst_ip_3", "dst_ip_2", "dst_ip_1", "dst_ip_0"]] = self.original_vals["Dst IP"].str.split(".", n=3, expand=True).astype(int)
            self.original_vals.drop(["Src IP", "Dst IP"], inplace=True, axis=1)

            # revied from (https://www.deeplearningnerds.com/pandas-add-columns-to-a-dataframe-copy/) but I knew .get_dummies() was possible before
            self.original_vals = pd.get_dummies(self.original_vals, columns=["Protocol"])

            # Picked parquet because of this article: https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d
            # It has best storage size
            self.original_vals.to_parquet(os.path.join(datasets_folder_path, f"ACI-IoT-2023-flow{group_str}-formatted-undersampled.parquet"))
        else:
            self.original_vals = pd.read_parquet(os.path.join(datasets_folder_path, f"ACI-IoT-2023-flow{group_str}-formatted-undersampled.parquet"))

        if not os.path.exists(os.path.join(datasets_folder_path, f"ACI-IoT-flow{group_str}-counts.csv")):
            self.original_vals["Label"].value_counts().to_csv(os.path.join(datasets_folder_path, f"ACI-IoT-flow{group_str}-counts.csv"))

        # Get the classes
        self.classes = {label: num for num, label in enumerate(self.original_vals["Label"].unique())}
        self.number_of_classes = len(self.classes)
        self.classes.update({num: label for label, num in self.classes.items()})
        self.original_vals["label"] = self.original_vals.pop("Label").map(self.classes)

        self.format = target_format

        # Scalers apparently work well with dataframes? https://stackoverflow.com/a/36475297
        self.scaler.fit(self.original_vals)
        self.dat = pd.DataFrame(self.scaler.transform(self.original_vals), columns=self.original_vals.columns)
        self.dat["label"] = self.original_vals["label"]

        self.number_of_features = len(self.original_vals.columns) - 2  # -1 extra for index
        self.feature_labels = {feature_num: column_name for feature_num, column_name in enumerate(filter(lambda x: x not in {"label", "index"}, self.original_vals.columns))}

        assert True not in pd.isna(self.original_vals)
        assert True not in pd.isna(self.dat)

        pass

    def __len__(self) -> int:
        return len(self.original_vals)

    def __getitem__(self, index: int) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        if self.scaler_status == 0:
            print("Warning, no scaler set! please set the scaler!")
            self.scaler_status = -1
        item = self.dat.iloc[index]
        item.pop("index")
        tar = item.pop("label")
        assert True not in pd.isna(item)
        features = torch.Tensor(item.astype(float).to_numpy())
        target = torch.Tensor(tar)
        target = target.long()
        target.requires_grad = False
        target = self.target_to_one_hot(target)
        return features, target

    def __getitems__(self, indexes: list[int]) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        if self.scaler_status == 0:
            print("Warning, no scaler set! please set the scaler!")
            self.scaler_status = -1
        items = self.dat.iloc[indexes]
        items.pop("index")
        tar = items.pop("label")
        assert True not in pd.isna(items)
        features = torch.Tensor(items.astype(float).to_numpy())
        # features.apply_(lambda x: self.scale.transform(x))
        targets = torch.Tensor(tar.astype(int).to_numpy())
        targets = targets.long()
        targets.requires_grad = False
        targets = self.target_to_one_hot(targets)
        return features, targets

    @staticmethod
    def group_labels(old_label):
        if "Flood" in old_label:
            return "Flood"
        if "Scan" in old_label:
            return "Scan"
        else:
            return old_label


class tabularBenchmark(BaseDataset):
    # https://huggingface.co/datasets/inria-soda/tabular-benchmark

    def __init__(self, target_format: str = "CrossEntropy"):
        super().__init__()
        if os.path.exists(os.path.join("datasets", "huggingface_dataset_cache.csv")):
            self.original_vals = pd.read_csv(os.path.join("datasets", "huggingface_dataset_cache.csv"))
        else:
            self.original_vals = pd.read_csv("hf://datasets/inria-soda/tabular-benchmark/clf_cat/albert.csv")
            self.original_vals.to_csv(os.path.join("datasets", "huggingface_dataset_cache.csv"))
        self.original_vals["label"] = self.original_vals.pop("class").astype(str)

        # Get the classes
        self.classes = {label: num for num, label in enumerate(self.original_vals["label"].unique())}
        self.number_of_classes = len(self.classes)
        self.classes.update({num: label for label, num in self.classes.items()})
        self.original_vals["label"] = self.original_vals["label"].map(self.classes)

        self.format = target_format
        # Scalers apparently work well with dataframes? https://stackoverflow.com/a/36475297
        self.scaler.fit(self.original_vals)
        self.dat = pd.DataFrame(self.scaler.transform(self.original_vals), columns=self.original_vals.columns)
        self.dat["label"] = self.original_vals["label"]

        self.number_of_features = len(self.original_vals.columns) - 1
        self.feature_labels = {feature_num: column_name for feature_num, column_name in enumerate(filter(lambda x: x not in {"label", "index"}, self.original_vals.columns))}

    def __len__(self) -> int:
        return len(self.original_vals)

    def __getitem__(self, index: int) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        if self.scaler_status == 0:
            print("Warning, no scaler set! please set the scaler!")
            self.scaler_status = -1
        item = self.dat.iloc[index]
        # item.pop("index")
        tar = item.pop("label")
        features = torch.Tensor(item.astype(float).to_numpy())
        target = torch.Tensor(tar)
        target = target.long()
        target.requires_grad = False
        target = self.target_to_one_hot(target)
        return features, target

    def __getitems__(self, indexes: list[int]) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
        if self.scaler_status == 0:
            print("Warning, no scaler set! please set the scaler!")
            self.scaler_status = -1
        items = self.dat.iloc[indexes]
        # items.pop("index")
        tar = items.pop("label")
        features = torch.Tensor(items.astype(float).to_numpy())
        # features.apply_(lambda x: self.scale.transform(x))
        targets = torch.Tensor(tar.astype(int).to_numpy())
        targets = targets.long()
        targets.requires_grad = False
        targets = self.target_to_one_hot(targets)
        return features, targets


def collate_fn_(items: tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor] | list[tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]]) -> tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]:
    """
    Function for joining key, feature tuples together in the format that we want it in.

    Args:
        items (tuple[InputFeatures  |  torch.Tensor, Targets  |  torch.Tensor] | list[tuple[InputFeatures  |  torch.Tensor, Targets  |  torch.Tensor]]): Items to join together.

    Returns:
        tuple[InputFeatures | torch.Tensor, Targets | torch.Tensor]: The items joined and then repackaged into  tuple.
    """
    if isinstance(items, tuple):
        items = [items]
    features, tags = zip(*items)
    return (torch.cat(features), torch.cat(tags))


def get_dataloader(config: ConfigObject | None = None, dataset: None | BaseDataset = None) -> ModifiedDataloader:
    """
    Gets the dataloader specified by the config or wraps a dataset in a dataloader. If neither is given the default config from cfg.py is used.

    Args:
        config (ConfigObject | None, optional): Config to specify what dataset to retrieve if none is given. Defaults to None.
        dataset (None | BaseDataset, optional): Dataset to wrap in a dataloader so that it can be iterated over. Defaults to None.

    Returns:
        torch.utils.data.DataLoader: torch dataloader that can be iterated over.
    """
    if dataset is None:
        if config is None:
            print("Config was not given for dataset, creating config")
            config = ConfigObject()
        dataset = get_dataset(config)
    dl: ModifiedDataloader = torch.utils.data.DataLoader(dataset, batch_size=config("BatchSize"), num_workers=config("NumberOfWorkers"), persistent_workers=True if config("NumberOfWorkers") > 1 else False, **dataset.base.load_kwargs)
    dl.base = dataset.base
    return dl


def get_dataset(config: ConfigObject) -> BaseDataset:
    """
    Retrieves the dataset specified by the config. Also sets the number of classes and number of features config parameters.

    Args:
        config (ConfigObject): Config to use to retrieve the correct dataset.

    Returns:
        BaseDataset: The dataset retrieved from the config.
    """
    datasets: dict[str, torch.utils.data.Dataset] = {"Vulnerability": VulnerabilityDataset, "RandomDummy": RandomDummyDataset, "ACI": ACIIOT2023, "ACI_grouped": partial(ACIIOT2023, grouped=True), "ACI_grouped_full_balance": partial(ACIIOT2023, grouped=True, difference_multiplier=1), "ACI_flows": ACIPayloadless, "Tabular": tabularBenchmark}
    data: BaseDataset = datasets[config("DatasetName")](target_format=config("LossFunction", getBaseForm=True))
    config("NumClasses", data.number_of_classes)
    config("NumFeatures", data.number_of_features)
    if config("MaxSamples") > 0:
        data, _ = torch.utils.data.random_split(data, [config("MaxSamples"), len(data) - config("MaxSamples")], generator=torch.Generator().manual_seed(1))
        data.base = data.dataset.base
    assert data.base.load_kwargs["collate_fn"] is collate_fn_
    return data


def get_train_test(config: ConfigObject | None = None, dataset: None | BaseDataset = None, dataloader: None | ModifiedDataloader = None) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset] | tuple[ModifiedDataloader, ModifiedDataloader]:
    """
    Splits a dataset or the data in a dataloader into train and test datasets by the percentages given in config.
    The first such split generates a Scaler for the features from the training half and applies it to both the train and test.
    If both dataloader and dataset are None it calls get_dataset(config) to get the dataset and returns the train/test split of that.

    Args:
        config (ConfigObject | None, optional): Config to retrieve the train/test percentage from. If None, it uses the default from cfg.py. Defaults to None.
        dataset (None | BaseDataset, optional): Dataset to split. Only give one of dataset or dataloader. Defaults to None.
        dataloader (None | torch.utils.data.DataLoader, optional): Dataloader to split, returns two dataloaders if this is used. Defaults to None.

    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset] | tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: returns either a pair of datasets that are exclusively split or a pair of dataloaders if dataloader is not None
    """
    if config is None:
        print("Config was not given for dataset, creating config")
        config = ConfigObject()
    if dataset is not None and dataloader is not None:
        print("Incorrect usage of get_train_test splitting, please only give a dataset OR a dataloader, not both.")
    elif dataset is not None:
        train, test = torch.utils.data.random_split(dataset, [1 - config("TrainTest"), config("TrainTest")], generator=torch.Generator().manual_seed(1))
        if dataset.base.scaler_status != 1:
            dataset.base.scale_indices(train.indices)
        train.base = dataset.base
        test.base = dataset.base
        return train, test
    elif dataloader is not None:
        train_ds, test_ds = get_train_test(config=config, dataset=dataloader.dataset)
        return get_dataloader(config=config, dataset=train_ds), get_dataloader(config=config, dataset=test_ds)
    else:
        return get_train_test(config, get_dataset(config))


def split_by_class(dataloader: ModifiedDataloader, classes_to_use: list[int], config: ConfigObject, individual=False) -> ModifiedDataloader | list[ModifiedDataloader]:
    data_lists = {x: list() for x in classes_to_use}
    for (X, y) in dataloader:
        pairs = zip(X, y)
        sorted_pairs = sorted(pairs, key=lambda x: x[1])
        groups = groupby(sorted_pairs, key=lambda x: x[1])
        for g_name, g in groups:
            if g_name.item() in data_lists.keys():
                (grouped_X, grouped_y) = zip(*g)
                data_lists[g_name.item()].extend(grouped_X)

    for item in classes_to_use:
        data_lists[item] = SingleClass(data_lists[item], item, dataloader.base)
        # print(f"{item}: {len(data_lists[item])}")

    if individual:
        dls = [get_dataloader(config, data_lists[x]) for x in sorted(list(data_lists.keys()))]
        for num, dl in enumerate(dls):
            dl.base = data_lists[num].base
        return dls

    dl = get_dataloader(config, torch.utils.data.ConcatDataset(data_lists.values()))
    dl.base = data_lists[list(data_lists.keys())[0]].base
    return dl
