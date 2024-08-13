import torch
from torch.utils.data import DataLoader
from numpy import ndarray
from thop import profile
from sklearn.metrics import f1_score
from . import cfg
from .extramodules import Nothing_Module


class ModelFunctions():
    def __init__(self):
        # Non-Overridden values (These actually store things)
        self.epoch_callbacks = []
        self.dataloader = None
        self.validation_dataloader = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.loss_fn = None
        self.loss_additive_info: tuple[callable, tuple] = torch.zeros, (1, )
        self.frozen = {}
        self.pruning_layers = []

        # Overriden Values (should be overriden by multi-inheritence)
        self.cfg = cfg.ConfigObject()
        # self.train = True
        # self.parameters = None  # <- Does not work as it overrides the actual function that should be there

    def set_training_data(self, dataloader: DataLoader | None = None) -> None:
        self.dataloader = dataloader

    def fit(self, epochs: int = 0, dataloader: DataLoader | None = None, keep_callbacks=False) -> str:
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

        if epochs == 0:
            self.train(False)
            epochs = 1

        self = self.to(self.cfg("Device"))
        for e in range(epochs):
            epoch_results = self.run_single_epoch(dl)
            if self.validation_dataloader is not None:
                val_epoch_results = {f"val_{x[0]}": x[1] for x in self.run_single_epoch(self.validation_dataloader).items()}
            else:
                val_epoch_results = {f"val_{x}": 0.0 for x in epoch_results.keys()}
            epoch_results = epoch_results | val_epoch_results

            epoch_results["epoch"] = e

            for call in self.epoch_callbacks:
                call(epoch_results)

        if not keep_callbacks:
            self.epoch_callbacks = []

        return f'Ran model with {epoch_results["f1_score"]:2.3f}% F1 on final epoch {e}'

    def run_single_epoch(self, dataloader) -> dict[str, float]:
        results = {"total_loss": 0, "f1_score": 0.0}
        results_of_predictions = {"True": [], "Predicted": []}
        for batch in dataloader:
            self.optimizer.zero_grad()
            self.zero_grad()
            X, y = batch
            X = X.to(self.cfg("Device"))
            y = y.to(self.cfg("Device"))
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

        results["f1_score"] = f1_score(results_of_predictions["True"], results_of_predictions["Predicted"], average="weighted")
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

    def get_FLOPS(self):
        macs, params = profile(self, inputs=(self.dataloader.dataset.__getitem__(0)[0].unsqueeze(dim=0).to(self.cfg("Device")), ))
        return macs

    def get_parameter_count(self):
        macs, params = profile(self, inputs=(self.dataloader.dataset.__getitem__(0)[0].unsqueeze(dim=0).to(self.cfg("Device")), ))
        return params

    def get_zero_weights(self):
        count = 0

        for name, param in self.named_parameters():
            if "weight" in name:
                count += (param.data == 0).sum().item()

        return count

    def get_model_structure(self, count_zeros=False):
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

    def load_model_state_dict_with_structure(self: torch.nn.Module, state_dict: dict):
        for name, weights in state_dict.items():
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

            if contained_by == old:
                print("Failed to find the actual path to the module")

            # # https://stackoverflow.com/a/7616959
            # obj_class = old.__class__
            # new = obj_class()

            if isinstance(old, Nothing_Module):
                # If a module has been deleted, this resets it
                contained_by.__setattr__(name.split(".")[-2], old.old[0])
                del old
                old = contained_by.__getattr__(name.split(".")[-2])

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

    def state_dict_of_layer_i(self: torch.nn.Module, layer_i):
        state_dict = {}
        idx = -1
        for name, param in self.named_parameters():
            if "weight" in name:
                idx += 1
                if idx == layer_i:
                    state_dict = state_dict | {name: param.data.clone()}

            if "bias" in name and idx == layer_i:
                state_dict = state_dict | {name: param.data.clone()}

        return state_dict

    def additive_loss(self, **kwargs):
        # Any additional terms to be added to the loss
        val = self.loss_additive_info[0](*(self.loss_additive_info[1]), **kwargs)
        return val
