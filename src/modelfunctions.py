import time
import torch
from torch.utils.data import DataLoader
from numpy import ndarray
from thop import profile
from sklearn.metrics import f1_score
from . import cfg
from .extramodules import Nothing_Module, PreMutablePruningLayer, PostMutablePruningLayer
from typing import Callable
import tqdm
import os


class ModelFunctions():
    def __init__(self):
        assert isinstance(self, torch.nn.Module)
        # Non-Overridden values (These actually store things)
        self.epoch_callbacks: list[Callable[[dict], None]] = []
        self.dataloader: None | DataLoader = None
        self.validation_dataloader: None | DataLoader = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.loss_fn: torch.nn.Module | None = None
        self.loss_additive_info: tuple[Callable, tuple] = torch.zeros, (1, )
        self.frozen: dict[str, torch.Tensor] = {}
        self.pruning_layers: list[PreMutablePruningLayer | PostMutablePruningLayer] = []

        # Overriden Values (should be overriden by multi-inheritence)
        self.cfg: cfg.ConfigObject = cfg.ConfigObject()
        # self.train = True
        # self.parameters = None  # <- Does not work as it overrides the actual function that should be there

    def set_training_data(self, dataloader: DataLoader | None = None) -> None:
        self.dataloader = dataloader

    def set_validation_data(self, dataloader: DataLoader | None = None) -> None:
        self.validation_dataloader = dataloader

    def get_progress_bar(self, epochs):
        if os.path.exists("results/progressbar.txt"):
            self.progres_file = open("results/progressbar.txt", mode="r+")  # This might be useful so I am putting it here: https://stackoverflow.com/a/72412819
            self.progres_file.seek(0, 2)
            progres_pos = self.progres_file.tell()
            progres_bar = tqdm.tqdm(range(epochs), desc=f"Fit \t|{self.cfg('PruningSelection')}| \tPID:{os.getpid()}\t", total=epochs, file=self.progres_file, ascii=True)
            self.progress_need_to_remove = []
            self.progress_need_to_remove.append(lambda r: self.progres_file.seek(progres_pos, 0))
            self.progress_need_to_remove.append(lambda results: progres_bar.set_postfix_str(f"{results['f1_score_macro']*100:2.3f}% Train F1, {results['val_f1_score_macro']*100:2.3f}% Validation F1"))
            self.progress_need_to_remove.append(lambda r: self.progres_file.seek(progres_pos, 0))
            self.epoch_callbacks.extend(self.progress_need_to_remove)

            return progres_bar
        else:
            return None

    def remove_progress_bar(self):
        for x in self.progress_need_to_remove:
            self.epoch_callbacks.remove(x)
        self.progres_file.close()

    def fit(self, epochs: int = 0, dataloader: DataLoader | None = None, keep_callbacks: bool = False) -> str:
        assert isinstance(self, torch.nn.Module)
        self: torch.nn.Module | ModelFunctions  # More typehint

        if dataloader is None:
            if self.dataloader is None:
                raise TypeError("No dataset selected for Automatic Vulnerability Detection training")
            dl = self.dataloader
        else:
            dl = dataloader

        # TODO: Configure the optimizer with optimizer specific kwargs
        if self.optimizer is None:
            self.optimizer = self.cfg("Optimizer")(self.parameters(), lr=self.cfg("LearningRate"))

        if self.loss_fn is None:
            self.loss_fn = self.cfg("LossFunction")()

        # If no epochs, assume no training
        if epochs == 0:
            self.train(False)
            with torch.no_grad():
                resultsmessage = self.fit(epochs=1, dataloader=dataloader, keep_callbacks=keep_callbacks)  # Run model to collect data still.
            return resultsmessage

        self = self.to(self.cfg("Device"))  # Move things to the active device

        frozen_bad = [x for x in self.frozen.keys() if (x not in self.state_dict().keys()) or (self.frozen[x].shape != self.state_dict()[x].shape)]
        for incorrect_frozen in frozen_bad:
            self.frozen.pop(incorrect_frozen)

        progres_bar = self.get_progress_bar(epochs)

        for e in progres_bar if progres_bar is not None else range(epochs):
            # Run the epoch
            epoch_results = self.run_single_epoch(dl)

            # Run the validation dataset if it exists
            if self.validation_dataloader is not None:
                mode = self.training
                self.train(False)
                # Validation data has the same name as normal data so it gets to be renamed
                val_epoch_results = {f"val_{x[0]}": x[1] for x in self.run_single_epoch(self.validation_dataloader).items()}
                self.train(mode)
            else:
                # If no validation data exists, just mark it as zeros
                val_epoch_results = {f"val_{x}": 0.0 for x in epoch_results.keys()}

            # Combine the two dictionaries
            epoch_results = epoch_results | val_epoch_results

            # Set current epoch number
            epoch_results["epoch"] = e

            # Run all of the callbacks
            for call in self.epoch_callbacks:
                call(epoch_results)

        if progres_bar is not None:
            # Clear out the tqdm callback
            self.remove_progress_bar()

        # Clear out old callbacks unless specified.
        if not keep_callbacks:
            self.epoch_callbacks = []

        # Just a quick message about the run
        return f'Ran model with {epoch_results["f1_score_macro"]*100:2.3f}% or {epoch_results["f1_score_weight"]*100:2.3f}% F1 on final epoch {e}'

    def run_single_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        assert isinstance(self, torch.nn.Module)
        self: torch.nn.Module | ModelFunctions  # More typehint

        results = {"total_loss": 0, "f1_score_weight": 0.0, "f1_score_macro": 0.0}
        results_of_predictions = {"True": [], "Predicted": []}

        for batch in dataloader:
            self.optimizer.zero_grad()
            self.zero_grad()
            X, y = batch
            X: torch.Tensor = X.to(self.cfg("Device"))
            y: torch.Tensor = y.to(self.cfg("Device"))
            y_predict = self(X)

            # print(y_predict)
            # print(y)

            loss: torch.Tensor = self.loss_fn(y_predict, y) + self.additive_loss()

            if self.training:
                loss.backward()
                self.optimizer.step()
                # TODO: Scheduler steps

            results["total_loss"] += loss.detach().item()

            results_of_predictions["True"].extend(y.detach().cpu())
            results_of_predictions["Predicted"].extend(y_predict.argmax(dim=1).detach().cpu())

            # Reset frozen weights
            self.load_state_dict(self.frozen, strict=False)

        results["f1_score_weight"] = f1_score(results_of_predictions["True"], results_of_predictions["Predicted"], average="weighted")
        results["f1_score_macro"] = f1_score(results_of_predictions["True"], results_of_predictions["Predicted"], average="macro")
        # [results_of_predictions["True"] != results_of_predictions["True"][0]]
        # results["max_class_removed_f1_score"] = f1_score(results_of_predictions["True"], results_of_predictions["Predicted"], average="weighted")
        results["mean_loss"] = results["total_loss"] / len(results_of_predictions["True"])

        return results

    def predict(self, inputs_: torch.Tensor | list[torch.Tensor] | ndarray) -> torch.Tensor | ndarray:
        if not isinstance(inputs_, torch.Tensor):
            with torch.no_grad():
                # This should handle lists of tensors and ndarrays, and then output as an ndarray
                inputs_tensor = torch.tensor(inputs_)
                outputs_tensor = self(inputs_tensor)
                return outputs_tensor.detach().numpy()

        # Check if you want the gradients
        if self.training:
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

    def get_FLOPS(self) -> int:
        macs, params = profile(self, inputs=(self.dataloader.dataset.__getitem__(0)[0].unsqueeze(dim=0).to(self.cfg("Device")), ))
        return int(macs)

    def get_parameter_count(self) -> int:
        macs, params = profile(self, inputs=(self.dataloader.dataset.__getitem__(0)[0].unsqueeze(dim=0).to(self.cfg("Device")), ))
        return int(params)

    def get_zero_weights(self) -> int:
        """
        Counts the number of weights in the model that are equal to zero
        """
        assert isinstance(self, torch.nn.Module)
        count = 0

        for name, param in self.named_parameters():
            if "weight" in name:
                count += (param.data == 0).sum().item()

        return count

    def get_model_structure(self, count_zeros: bool = False) -> str:
        """
        finds the structure of the weights, when flattned. If count_zeros is true, weights that are zero are included in this final count.
        """
        assert isinstance(self, torch.nn.Module)
        counts = ""

        for name, param in self.named_parameters():
            if "weight" in name:
                if count_zeros:
                    counts = counts + str((param.data.sum(dim=0) != 0).sum().item() + (param.data.sum(dim=0) == 0).sum().item()) + "/"
                    counts = counts + str((param.data.sum(dim=1) != 0).sum().item() + (param.data.sum(dim=1) == 0).sum().item()) + ", "
                else:
                    counts = counts + str((param.data.sum(dim=0) != 0).sum().item()) + "/"
                    counts = counts + str((param.data.sum(dim=1) != 0).sum().item()) + ", "

        return counts[:-2]

    def load_model_state_dict_with_structure(self: torch.nn.Module, state_dict: dict[str, torch.Tensor]):
        """
        Loads the model from a state dict that is a paired down version of the model.
        """
        assert isinstance(self, torch.nn.Module)

        for name, weights in state_dict.items():
            if "total" in name:
                # Skip any of these "total" values, no clue where they come from.
                continue
            name: str
            weights: torch.Tensor

            # if name not in self.state_dict().keys() and "Active_check" not in name:
            #     # The Active check is just a flag to make sure that this function reactivates the module
            #     print("This function can only load state dictionaries like the ones in the model structure")

            contained_by = self
            old = self
            for module in name.split(".")[:-1]:
                contained_by = old
                old = old.__getattr__(module)
                if isinstance(old, Nothing_Module):
                    # If a module has been deleted, this resets it
                    contained_by.__setattr__(module, old.old[0])
                    del old
                    old = contained_by.__getattr__(module)

            if contained_by == old:
                print("Failed to find the actual path to the module")

            # # https://stackoverflow.com/a/7616959
            # obj_class = old.__class__
            # new = obj_class()

            if "weight" in name.split(".")[-1]:
                shape = weights.shape
                if isinstance(old, torch.nn.Linear):
                    contained_by.__setattr__(name.split(".")[-2], torch.nn.Linear(shape[1], shape[0]))
                elif isinstance(old, torch.nn.Conv1d):
                    contained_by.__setattr__(name.split(".")[-2], torch.nn.Conv1d(shape[1], shape[0], shape[2]))
                else:
                    print("Module not yet able to be replaced")

            if "_extra_state" in name.split(".")[-1] and weights in "NONE":
                contained_by.__setattr__(name.split(".")[-2], Nothing_Module(contained_by.__getattr__(name.split(".")[-2])))

        self.load_state_dict(state_dict=state_dict, strict=False)

    def state_dict_of_layer_i(self: torch.nn.Module, layer_i: int) -> dict[str, torch.Tensor]:
        """
        Gets the state dict of the module at position i.
        """

        for name, module in self.named_modules():
            if module is self.get_important_modules()[layer_i]:
                state_dict = {f"{name}.{x}": y for x, y in module.state_dict().items()}

        return state_dict

    def additive_loss(self, **kwargs) -> torch.Tensor:
        # Any additional terms to be added to the loss
        val = self.loss_additive_info[0](*(self.loss_additive_info[1]), **kwargs).to(self.cfg("Device"))
        return val

    def save_model_state_dict(self, logger: None | Callable, name: str | None = None, update_config: bool = True, logger_column: None | str = None):
        assert isinstance(self, torch.nn.Module)
        if logger_column is None:
            loggercolumn = "SaveLocation"
        else:
            loggercolumn = logger_column

        if update_config:
            if logger_column is not None:
                print("Savepoint is being changed in the config but not in the log, are you sure you want to do this?")
            # this is the default, where it saves the name as the config savelocation value
            if name is None:
                if self.cfg("SaveLocation") is None:
                    self.cfg("SaveLocation", f"ModelStateDict{time.time()}.pt")
            else:
                self.cfg("SaveLocation", name if ".pt" in name else f"{name}.pt")
            logger(loggercolumn, self.cfg("SaveLocation"))
            torch.save(self.state_dict(), "savedModels/"+self.cfg("SaveLocation"))
        else:
            # This is if you want an extra save (making it for waypoints)
            name = f"ModelStateDict{logger_column if logger_column is not None else ''}{time.time()}.pt" if name is None else (name if ".pt" in name else f"{name}.pt")
            logger(loggercolumn, name)
            torch.save(self.state_dict(), "savedModels/"+name)
