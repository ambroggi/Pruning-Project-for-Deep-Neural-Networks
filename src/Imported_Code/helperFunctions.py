import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.modelstruct import BaseDetectionModel
    BaseDetectionModel


class ConfigCompatabilityWrapper():
    def __init__(self, config, translations="ADDM"):
        self.config = config
        if translations == "ADDM":
            self.translations = {
                "num_epochs": "NumberOfEpochs",
                "lr": "LearningRate",
                "num_pre_epochs": "NumberOfEpochs",
                "alpha": "AlphaForADMM",
                "rho": "RhoForADMM",
                "k": "LayerPruneTargets",
                "percent": "WeightPrunePercent"
            }
        elif translations == "TOFD":
            self.translations = {
                "alpha": "AlphaForTOFD",
                "beta": "BetaForTOFD",
                "t": "tForTOFD"
            }
        else:
            self.translations = {}

    def __getattr__(self, name: str):
        if name == "translations":
            return super().__getattr__(name)
        if name in ["l2"]:
            return True
        if name in ["l1"]:
            return False

        return self.config(self.translations.get(name, name))


class forward_hook():
    def __init__(self):
        self.inp: None | torch.Tensor = None
        self.out: None | torch.Tensor = None
        self.out_no_bias: None | torch.Tensor = None

    def __call__(self, module: torch.nn.Module, inp: torch.Tensor, out: torch.Tensor):
        self.modu = module
        if hasattr(module, "bias") and module.bias is not None:
            self.inp = inp[0]
            self.out = out
            bias = module.bias
            module.bias = None
            module(inp[0])
            module.bias = bias
        else:
            self.inp = inp[0]
            self.out_no_bias = out
            if self.out is None:
                self.out = out
        # print(f"input: {inp}")


def remove_layers(model: "BaseDetectionModel", parameter_to_reduce: int, keepint_tensor, length_of_single_channel=1) -> dict[str, torch.Tensor]:
    """
    Reduces a layer (selected by its index (parameter_to_reduce)) using keepint_tensor, a tensor of bools.
    This is used for ThiNet

    The code is not great as this was made earlier in the project
    """

    state_dict = {}
    reduced_module, next_module = ((name, module) for name, module in model.named_modules() if module in model.get_important_modules()[parameter_to_reduce: parameter_to_reduce+2])
    for name, param in reduced_module[1].named_parameters():
        name = reduced_module[0] + "." + name
        state_dict.update({name: param.data[keepint_tensor[:: length_of_single_channel]]})
    for name, param in next_module[1].named_parameters():
        name = next_module[0] + "." + name
        if "weight" in name:
            if len(keepint_tensor) != len(param.data[0]):
                # This is for convolutional channels, when flattened each channel crosses many fully connected
                # So I try to flatten the keeping tensor while making sure that all items from the correct channels are kept
                multiplier = len(param.data[0])//len(keepint_tensor)
                t2 = torch.zeros(len(param.data[0]), dtype=torch.bool)
                for i in range(len(keepint_tensor)):
                    t2[i*multiplier:(i+1)*multiplier] = keepint_tensor[i]
                state_dict = state_dict | {name: param.data[:, t2]}

            else:
                state_dict = state_dict | {name: param.data[:, keepint_tensor]}
        else:
            state_dict.update({name: param.data})

    # This is actually replacing the model layers
    for name, values in state_dict.items():
        if name.split(".")[-1] == "weight":
            shape = values.shape
            m = model
            for x in name.split(".")[:-2]:
                m = m.__getattr__(x)
            old = m.__getattr__(name.split(".")[-2])
            if isinstance(old, torch.nn.Linear):
                m.__setattr__(name.split(".")[-2], torch.nn.Linear(shape[1], shape[0]))
            elif isinstance(old, torch.nn.Conv1d):
                m.__setattr__(name.split(".")[-2], torch.nn.Conv1d(shape[1], shape[0], old.kernel_size))

    # Then set the parameters
    model.load_state_dict(state_dict=state_dict, strict=False)
    model.to(model.cfg("Device"))
    return state_dict


def get_layer_by_state(model: torch.nn.Module, state_dict_key: str) -> torch.nn.Module:
    """
    Use a state dictionary to find the specific Module.
    """
    if state_dict_key.split(".")[-1] in ["weight", "bias"]:
        pth = state_dict_key.split(".")[:-1]
    else:
        pth = state_dict_key.split(".")

    old = model
    for p in pth:
        old = old.__getattr__(p)

    return old


def set_layer_by_state(model: torch.nn.Module, state_dict_key: str, obj):
    """
    Set a layer module by using a state dictionary (specifically the state dictionary key path)
    """
    if state_dict_key.split(".")[-1] in ["weight", "bias"]:
        pth = state_dict_key.split(".")[:-1]
    else:
        pth = state_dict_key.split(".")

    contained_by = model
    old = contained_by
    for p in pth:
        contained_by = old
        old = old.__getattr__(p)

    contained_by.__setattr__(p, obj)
