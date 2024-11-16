import os
import time
from typing import Callable

import torch
import tqdm
from numpy import ndarray
from sklearn.metrics import f1_score
from thop import profile
from torch.utils.data import DataLoader

from . import cfg
from .extramodules import (Nothing_Module, PostMutablePruningLayer,
                           PreMutablePruningLayer)


class ModelFunctions():
    """
    This is an abstract class, intended to be inhearited from to pass on all of the useful functions.
    """
    def __init__(self):
        assert isinstance(self, torch.nn.Module)
        # Non-Overridden values (These actually store things)
        self.epoch_callbacks: list[Callable[[dict], None]] = []  # Called at the end of each epoch with a dictionary of results. Is reset when the last epoch ends
        self.dataloader: None | DataLoader = None  # The labled data to load
        self.validation_dataloader: None | DataLoader = None  # The data that is to be used to verify the model performance
        self.optimizer: torch.optim.Optimizer | None = None  # Method of applying backpropigation
        self.loss_fn: torch.nn.Module | None = None  # This is the specific loss function that the model is using
        self.loss_additive_info: tuple[Callable, tuple] = torch.tensor, (0, )  # This is a space for any additional loss functions the model might have. Used with DAIS
        self.frozen: dict[str, torch.Tensor] = {}  # This is supposed to be a state dictionary for the current modules that are done with pruning. but it does not work that well
        self.pruning_layers: list[PreMutablePruningLayer | PostMutablePruningLayer] = []  # The currently being processed pruning layers
        self.make_progressbar = True  # wether or not to use a progress bar to show progression
        self.num_epochs_trained = 0  # Just to keep track of the number of epochs that were used to train the model

        # Overriden Values (should be overriden by multi-inheritence)
        self.cfg: cfg.ConfigObject = cfg.ConfigObject()
        # self.train = True
        # self.parameters = None  # <- Does not work as it overrides the actual function that should be there

    def set_training_data(self, dataloader: DataLoader | None = None) -> None:
        """A setter for the taining dataset

        Args:
            dataloader (DataLoader | None, optional): Pytorch dataloader to set as default training data. Defaults to None.
        """
        self.dataloader = dataloader

    def set_validation_data(self, dataloader: DataLoader | None = None) -> None:
        """A setter for the validation dataset

        Args:
            dataloader (DataLoader | None, optional): Python dataloader to set as the default validation data. Defaults to None.
        """
        self.validation_dataloader = dataloader

    def get_progress_bar(self, epochs: int) -> None | tqdm.tqdm:
        """This returns either a tqdm iterator that displays a prpgress bar in "results/progressbar.txt" if it exists or the console if it does not OR returns None if self.make_progressbar is false
        Also, this sets up for the progressbar to be removed later.

        Args:
            epochs (int): Number epochs to be expected for the progress bar.

        Returns:
            None | tqdm.tqdm: None if no progress bar was created, otherwise a tqdm object that can be iterated through.
        """
        if os.path.exists("results/progressbar.txt") and self.make_progressbar:
            # Write into the progressbar file, but this is a bit complicated because we want to keep overwriting the same line so that it shows progress correctly.
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
        elif self.make_progressbar:
            # if the file does not exist print to console. This is much easier because it is the default.
            self.progress_need_to_remove = []
            progres_bar = tqdm.tqdm(range(epochs), desc=f"Fit |{self.cfg('PruningSelection')}| PID:{os.getpid()}", total=epochs)
            self.progress_need_to_remove.append(lambda results: progres_bar.set_postfix_str(f"{results['f1_score_macro']*100:2.3f}% Train F1, {results['val_f1_score_macro']*100:2.3f}% Validation F1"))
            self.epoch_callbacks.extend(self.progress_need_to_remove)
            return progres_bar
        else:
            # disabled progressbar
            return None

    def remove_progress_bar(self):
        """
        Removes the progressbar if it exists and cleans them out of the epoch hooks.
        """
        for x in self.progress_need_to_remove:
            self.epoch_callbacks.remove(x)
        if len(self.progress_need_to_remove) > 1:
            self.progres_file.close()
        self.progress_need_to_remove = []

    def fit(self, epochs: int = 0, dataloader: DataLoader | None = None, keep_callbacks: bool = False) -> str:
        """This is the main function of training for the model. It sets things up based on the config if they were not alrady generated or given in the args
        Then it runs run_single_epoch per epoch interspaced with checks to the validation dataloader if it exist and running the callbacks.
        Returns a string stating the final results of the model and clears out the callbacks for future work (unless keep_callbacks is true).

        Use callbacks to gather data from the model, as it does not return anything useful here.

        Args:
            epochs (int, optional): Number of times to run through the dataset if it is zero it runs once with no training. Defaults to 0.
            dataloader (DataLoader | None, optional): The pytorch dataloader to use for each epoch. If None it attempts to load the default dataloader. Defaults to None.
            keep_callbacks (bool, optional): Check to keep callbacks from one fit over to the next. Most of the time callbacks are not kept and need to be remade. Defaults to False.

        Raises:
            TypeError: Raises a type error if dataloader is None and there is no default dataloader to load.

        Returns:
            str: String such as "Ran model with 2.4% or 7.1% F1 on with number of epochs 100" Where the first percentage is the macro f1 score and the second is weighted f1 score.
        """
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
        if epochs == 0 or (epochs != 1 and not self.training):
            self.train(False)
            self.num_epochs_trained -= 1  # Just to account for the repeat
            with torch.no_grad():
                resultsmessage = self.fit(epochs=1, dataloader=dataloader, keep_callbacks=keep_callbacks)  # Run model to collect data still.
            return resultsmessage

        self = self.to(self.cfg("Device"))  # Move things to the active device

        # Check that the frozen modules actually exist before applying them. (this is only really a problem when stacking pruning methods including BERT_Theseus)
        frozen_bad = [x for x in self.frozen.keys() if (x not in self.state_dict().keys()) or (self.frozen[x].shape != self.state_dict()[x].shape)]
        for incorrect_frozen in frozen_bad:
            self.frozen.pop(incorrect_frozen)

        progres_bar = self.get_progress_bar(epochs)

        # Get the scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 30) if self.cfg("SchedulerLR") == 1 else None

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

            scheduler.step() if (scheduler is not None) and self.training else None

            self.num_epochs_trained += 1

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
        return f'Ran model with {epoch_results["f1_score_macro"]*100:2.3f}% or {epoch_results["f1_score_weight"]*100:2.3f}% F1 on with number of epochs {self.num_epochs_trained}'

    def run_single_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Single step of the fit function this runs through the whole dataloader once per call. Returns a dictionary of all of the notable results

        Args:
            dataloader (DataLoader): Pytorch dataloader to run through the model

        Returns:
            dict[str, float]: dictionary of results from the model, should include:
            - total_loss: sum total of the loss for this run
            - f1_score_weight: The weighted F1 score for the run
            - f1_score_macro: The macro F1 score for the run
            - additive_loss: The total extra loss added by any additive loss functions to the model
            - f1_scores_all: list of f1 scores for each class
            - mean_loss: The total loss averaged over the number of items in the dataloader
        """
        assert isinstance(self, torch.nn.Module)
        self: torch.nn.Module | ModelFunctions  # More typehint

        results = {"total_loss": 0, "f1_score_weight": 0.0, "f1_score_macro": 0.0, "additive_loss": 0.0, "f1_scores_all": []}
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

            loss: torch.Tensor = self.loss_fn(y_predict, y)  # This is the normal loss function as defined by the config
            additive_loss: torch.Tensor = self.additive_loss()  # This is for algorithms that want to add extra loss measures
            loss += additive_loss

            if self.training:
                loss.backward()
                self.optimizer.step()
                # TODO: Scheduler steps

            results["total_loss"] += loss.detach().item()
            results["additive_loss"] += additive_loss.detach().item()

            results_of_predictions["True"].extend(y.detach().cpu() if y.ndim == 1 else y.argmax(dim=1).detach().cpu())
            results_of_predictions["Predicted"].extend(y_predict.argmax(dim=1).detach().cpu())

            # Reset frozen weights
            self.load_state_dict({name: torch.where(torch.isnan(value), self.state_dict()[name], value) for name, value in self.frozen.items()}, strict=False)
            assert False not in [False not in (torch.eq(value, self.state_dict()[name]) | torch.isnan(value)) for name, value in self.frozen.items()]

        results["f1_score_weight"] = f1_score(results_of_predictions["True"], results_of_predictions["Predicted"], average="weighted")
        results["f1_score_macro"] = f1_score(results_of_predictions["True"], results_of_predictions["Predicted"], average="macro")
        results["f1_scores_all"] = f1_score(results_of_predictions["True"], results_of_predictions["Predicted"], average=None, labels=range(self.cfg("NumClasses")), zero_division=0).tolist()
        # [results_of_predictions["True"] != results_of_predictions["True"][0]]
        # results["max_class_removed_f1_score"] = f1_score(results_of_predictions["True"], results_of_predictions["Predicted"], average="weighted")
        results["mean_loss"] = results["total_loss"] / len(results_of_predictions["True"])

        return results

    def predict(self, inputs_: torch.Tensor | list[torch.Tensor] | ndarray) -> torch.Tensor | ndarray:
        """Forgotten about function that predicts a list of output logits from a list of features. This is not used anywhere.

        Args:
            inputs_ (torch.Tensor | list[torch.Tensor] | ndarray): Input vector, tensor, or tensor list to be used to calculate classifications

        Returns:
            torch.Tensor | ndarray: output classifications from the model
        """
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
        """This is a dummy function that should not be able to be called. It is just here for typechecking so that it is clear what should be output from the model.
        In the actual case, this is overriden by torch.nn.Module's call function which is a wrapped version of forwards()

        Args:
            value (torch.Tensor): Tensor to apply model pipeline to.

        Returns:
            torch.Tensor: Output logits from the model.
        """
        print("Model Functions should be behind Module in the inheritence for the model class.")
        assert False
        return torch.tensor([0])

    def get_macs(self) -> int:
        """Returns the macs which should be a calculation of how many matrix multiplication calls there were in the model.

        Returns:
            int: Number representing the number of matrix multiplication calls there were in the model
        """
        macs, params = profile(self, inputs=(self.dataloader.dataset.__getitem__(0)[0].unsqueeze(dim=0).to(self.cfg("Device")), ), verbose=False)
        return int(macs)

    def get_parameter_count(self) -> int:
        """ Returns the number of parameters in the model

        Returns:
            int: Number of parameters found in the model
        """
        macs, params = profile(self, inputs=(self.dataloader.dataset.__getitem__(0)[0].unsqueeze(dim=0).to(self.cfg("Device")), ), verbose=False)
        return int(params)

    def get_zero_weights(self) -> int:
        """Counts the number of weights in the model that are equal to zero

        Returns:
            int: Number of zeroed out parameters in the model
        """
        assert isinstance(self, torch.nn.Module)
        count = 0

        for name, param in self.named_parameters():
            if "weight" in name:
                count += (param.data == 0).sum().item()

        return count

    def get_zero_filters(self) -> tuple[int, int]:
        """Counts the number of filters that are zero in the model.
        It counts both input and output where input is the number of filters from the prior layer that are ignored
        And output is the number of filters that only output zero (or the sum of biases as just the weights are counted.)

        Returns:
            tuple[int, int]: The count input and count output total for filters.
        """
        assert isinstance(self, torch.nn.Module)
        count_input = 0
        count_output = 0

        for name, param in self.named_parameters():
            if "weight" in name:
                count_input += (param.data.sum(dim=0) == 0).sum().item()
                count_output += (param.data.sum(dim=1) == 0).sum().item()

        return count_input, count_output

    def get_model_structure(self, count_zeros: bool = False) -> str:
        """Finds the structure of the weights, when flattned. If count_zeros is true, weights that are zero are included in this final count.

        Args:
            count_zeros (bool, optional): Wether or not to count pruned filters as well as regular filters. Defaults to False.

        Returns:
            str: Model structure in string format such as "10/20, 20/5" where there are 10 initial features that get expanded to 20 and then reduced to 5
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
        """Loads the model from a state dict that is a paired down version of the model.

        Args:
            state_dict (dict[str, torch.Tensor]): A state dictionary full of tensors that corrispond to the diffrent model layers.
        """
        assert isinstance(self, torch.nn.Module)
        self.optimizer = None
        self.num_epochs_trained = 0
        self.frozen = {}

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
        """Gets the state dict of the module at position i.

        Args:
            layer_i (int): Number layer from important modules (those that can be pruned). So 0 is the first prunable module and 1 is the second and so forth.

        Returns:
            dict[str, torch.Tensor]: The state dictionary of the module.
        """

        for name, module in self.named_modules():
            if module is self.get_important_modules()[layer_i]:
                state_dict = {f"{name}.{x}": y for x, y in module.state_dict().items()}
                return state_dict

        return state_dict

    def additive_loss(self, **kwargs) -> torch.Tensor:
        """This applies whatever extra loss function you put in loss_additive_info. It is used for algorithms that add an additional loss metric.

        Returns:
            torch.Tensor: The extra loss as a torch tensor (assuming the loss_additive_info also returns a torch tensor)
        """
        # Any additional terms to be added to the loss
        val = self.loss_additive_info[0](*(self.loss_additive_info[1]), **kwargs).to(self.cfg("Device"))
        return val

    def save_model_state_dict(self, logger: None | Callable, name: str | None = None, update_config: bool = True, logger_column: None | str = None):
        """Save the model state dictionary as a .pt (PyTorch) file for later retrival.

        Args:
            logger (None | Callable): This is a function that takes "SaveLocation" or logger_column and the name of the save file as its two arguments. Mostly useful for loggging the save location.
            name (str | None, optional): Name of the save file. Will generate a new name if this is none. Defaults to None.
            update_config (bool, optional): Updates the config to say that the model is stored in "SaveLocation". This is useful for the main model version. Defaults to True.
            logger_column (None | str, optional): Override for "SaveLocation" as first argument for logger. Defaults to None.
        """
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
