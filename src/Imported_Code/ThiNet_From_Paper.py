# This is an implementation of ThiNet from the paper: ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression (https://arxiv.org/pdf/1707.06342)
# Implementation made by Alexandre Broggi 2024, I hope I am not making any big mistakes
import torch
import torch.utils.data
from .. import modelstruct


def thinet_pruning(model: torch.nn.Module, parameterNumber: int, config, dataset: torch.utils.data.DataLoader | torch.utils.data.Dataset | None = None):

    with torch.no_grad():
        if dataset is None:
            if hasattr(model, "dataloader"):
                dataset = model.dataloader
            else:
                raise TypeError("Missing positional arguement 'dataset'")
        if isinstance(dataset, torch.utils.data.DataLoader):
            dataset = dataset.dataset

        input_storage = forward_hook()

        i = 0
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                if parameterNumber + 1 == i:
                    # Prepare to catch the data for the module
                    removable = module.register_forward_hook(input_storage)
                    break
                elif parameterNumber == i:
                    module_being_reduced = module
                i += 1
        else:
            # This runs if the break is not hit
            raise IndexError("thinet cannot be applied to the last layer")

        # Create sample of dataset, Fancy load_kwargs is just there to load the collate_fn
        training_data = iter(torch.utils.data.DataLoader(dataset, 100000, **(dataset.load_kwargs if hasattr(dataset, "load_kwargs") else {}))).__next__()

        # Run data through the model
        model(training_data[0])
        removable.remove()

        # Old code that checked if the model output is how it is expected to be.
        # expected_output = module(input_storage.inp)
        # print(f"Sum = {sum((expected_output - input_storage.out))}")

        # pruning_mask = torch.zeros(len(input_storage.inp[0]))
        # removed_bias = torch.ones(len(input_storage.out[0]))

        remove_handle = modelstruct.PreSoftPruningLayer(module)
        remove_handle.para.data = torch.zeros_like(remove_handle.para.data)
        pruning_mask = remove_handle.para.data

        # If moving from convolutional to linear layers channels can cover multiple filters
        length_of_single_channel = 1
        if isinstance(module_being_reduced, torch.nn.Conv1d):
            length_of_single_channel = len(pruning_mask) // module_being_reduced.out_channels

        # ThiNetPruning.apply(module, "weight", set_called_T=pruning_mask)
        # ThiNetPruning.apply(module, "bias", set_called_T=removed_bias)

        # Removing the bias
        bias_storage = module.bias.data
        module.bias.data = torch.zeros_like(bias_storage)

        # This is Algorithm 1 from the paper.
        C = len(pruning_mask) // length_of_single_channel  # Number of channels
        r = config("WeightPrunePercent")[parameterNumber]  # Percent to remain in this layer
        removed_indexes = []

        # While we have not pruned enough channels
        while len(removed_indexes) < (1-r)*C:
            # Set the found minimum to be infinity and the channel to be None
            min_value = torch.inf
            min_i = None

            # Go through all possible channels
            for i in range(C):
                if i not in removed_indexes:  # That have not been removed from "I" yet
                    removed_indexes.append(i)
                    pruning_mask[i*length_of_single_channel:(i+1)*length_of_single_channel] = 1

                    # Recreating equation 6
                    x: torch.Tensor = module(input_storage.inp)
                    x = x.norm(2, 1)  # Technically I should square this but we are just checking the minimum, so the exact value does not matter
                    x = x.sum()

                    if x < min_value:
                        min_value = x.item()
                        min_i = i

                    pruning_mask[i*length_of_single_channel:(i+1)*length_of_single_channel] = 0
                    removed_indexes.pop(-1)

            removed_indexes.append(min_i)
            pruning_mask[min_i*length_of_single_channel:(min_i+1)*length_of_single_channel] = 1

        module.bias.data = bias_storage

        keeping_indexes = [a for a in range(C) if a not in removed_indexes]
        keepint_tensor = torch.BoolTensor([False for _ in range(len(pruning_mask))])
        for offset in range(length_of_single_channel):  # Just getting all of the weights associated with the removed channels
            keepint_tensor[[x + offset for x in keeping_indexes]] = True

        remove_handle.remove(update_weights=False)

        state_dict = {}

        idx = -1
        for names, params in model.named_parameters():
            if "weight" in names:
                idx += 1
            if idx == parameterNumber + 1:
                state_dict = state_dict | {f"{names.split('.')[0]}.{name}": (a[:, keepint_tensor] if name not in ["bias"] else a) for name, a in module.state_dict().items()}
            if idx == parameterNumber:
                state_dict = state_dict | {f"{names.split('.')[0]}.{name}": (a[keepint_tensor[:: length_of_single_channel]]) for name, a in module_being_reduced.state_dict().items()}

        # print(f"Module i+1 {module.named_parameters().__next__()[0]}, module i {module_minus1.named_parameters().__next__()[0]}")
        # print("Done")
        # test = {a: x.shape for a, x in model.state_dict().items()}
        # test2 = {a: x.shape for a, x in state_dict.items()}

        # This is actually replacing the model layers
        for name, values in state_dict.items():
            if name.split(".")[-1] == "weight":
                shape = values.shape
                old = model.__getattr__(name.split(".")[-2])
                if isinstance(old, torch.nn.Linear):
                    model.__setattr__(name.split(".")[-2], torch.nn.Linear(shape[1], shape[0]))
                elif isinstance(old, torch.nn.Conv1d):
                    model.__setattr__(name.split(".")[-2], torch.nn.Conv1d(shape[1], shape[0], old.kernel_size))

        # Then set the parameters
        model.load_state_dict(state_dict=state_dict, strict=False)
        return state_dict


class forward_hook():
    def __init__(self):
        self.inp: None | torch.Tensor = None
        self.out: None | torch.Tensor = None

    def __call__(self, module: torch.nn.Module, inp: torch.Tensor, out: torch.Tensor):
        self.inp = inp[0]
        self.out = out
        # print(f"input: {inp}")

# TODO: Need to figure out Equation 7
# It seems like the authors consider it to be optional from Figure 4, but I should still include it if possible.
