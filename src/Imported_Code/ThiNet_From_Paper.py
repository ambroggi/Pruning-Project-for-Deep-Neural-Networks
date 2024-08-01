# This is an implementation of ThiNet from the paper: ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression (https://arxiv.org/pdf/1707.06342)
# Implementation made by Alexandre Broggi 2024, I hope I am not making any big mistakes
import torch
import torch.utils.data
import modelstruct


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

        for i, module in enumerate(model.modules(), start = -1):
            if parameterNumber == i:
                # Prepare to catch the data for the module
                removable = module.register_forward_hook(input_storage)
                break
            elif parameterNumber == i-1:
                module_minus1 = module

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

        remove_handle = modelstruct.SoftPruningLayer(module)
        pruning_mask = remove_handle.para.data
        # ThiNetPruning.apply(module, "weight", set_called_T=pruning_mask)
        # ThiNetPruning.apply(module, "bias", set_called_T=removed_bias)

        # This is Algorithm 1 from the paper.
        C = len(pruning_mask)  # Number of channels
        r = config("WeightPrunePercent")[parameterNumber]  # Percent to prune from this layer
        removed_indexes = []

        # While we have not pruned enough channels
        while len(removed_indexes) < (1-r)*C:
            # Set the found minimum to be infinity and the channel to be None
            min_value = torch.inf
            min_i = None

            # Go through all possible channels
            for i in range(len(pruning_mask)):
                if i not in removed_indexes:  # That have not been removed from "I" yet
                    removed_indexes.append(i)
                    pruning_mask[i] = 1

                    # Recreating equation 6
                    x: torch.Tensor = module(input_storage.inp)
                    x = x.norm(2, 1)  # Technically I should square this but we are just checking the minimum, so the exact value does not matter
                    x = x.sum()

                    if x < min_value:
                        min_value = x.item()
                        min_i = i

                    pruning_mask[i] = 0
                    removed_indexes.pop(-1)

            removed_indexes.append(min_i)

        print("Done")


class forward_hook():
    def __init__(self):
        self.inp: None | torch.Tensor = None
        self.out: None | torch.Tensor = None

    def __call__(self, module: torch.nn.Module, inp: torch.Tensor, out: torch.Tensor):
        self.inp = inp[0]
        self.out = out
        # print(f"input: {inp}")

# Need to figure out Equation 7