# This is an implementation of BERT_Theseus from the paper: BERT-of-Theseus: Compressing BERT by Progressive Module Replacing (https://arxiv.org/pdf/2002.02925)
# Implementation made by Alexandre Broggi 2024, I hope I am not making any big mistakes
# Most of this is just to get things into the format that I want it to be in, only a few lines are actually directly related to the paper
import torch
from ..extramodules import Nothing_Module


class Theseus_Replacement(torch.nn.Module):

    def __init__(self, module_list: list[torch.nn.Module], input_shape, output_shape, model):
        super().__init__()
        self.replacing = torch.nn.ModuleList(module_list)
        self.flatten = None

        if len(input_shape) > 1 and len(output_shape) > 1:
            self.main = torch.nn.Conv1d(input_shape[0], output_shape[0], input_shape[1]-output_shape[1]+1)
        elif len(input_shape) > 1:
            self.main = torch.nn.Conv1d(input_shape[0], output_shape[0], output_shape[0])
            self.flatten = [flat_layer for flat_layer in module_list if isinstance(flat_layer, torch.nn.Flatten)][0]
        elif len(output_shape) > 1:
            print("Not sure how this is supposed to work, going from linear to Convolutional")
        else:
            self.main = torch.nn.Linear(input_shape[0], output_shape[0])

        self.rm1 = module_list[0].register_forward_pre_hook(lambda module, args: self.get_start_replacement(module, args))
        self.rm2 = module_list[-1].register_forward_hook(lambda module, args, output: self.get_end_replacement(module, args, output))

        self.b = model.cfg("BERTTheseusStartingLearningRate")
        self.k = model.cfg("BERTTheseusLearningRateModifier")/model.cfg("NumberOfEpochs")
        self.p = model.cfg("BERTTheseusStartingLearningRate")
        self.r = None
        self.output = None

        model.epoch_callbacks.append(lambda results: self.update_p(results["epoch"]))
        self.main.to(model.cfg("Device"))

    def get_start_replacement(self, module: torch.nn.Module, args: torch.Tensor):
        self.r = torch.bernoulli(self.p * torch.ones(len(args[0]), device=args[0].device)).unsqueeze(-1)
        self.output = self.main(args[0])
        if self.flatten is not None:
            self.output = self.flatten(self.output)

        while self.r.ndim < self.output.ndim:
            self.r = self.r.unsqueeze(-1)

    def get_end_replacement(self, module, args, output):
        # Equation 3 from the paper split into parts
        part1 = self.r * self.output
        part2 = (1 - self.r) * output

        # Test for failure:
        for x, y in zip(part1, part2):
            assert (torch.sum(x) == 0 or torch.sum(y) == 0)

        self.r = None
        self.output = None

        return part1 + part2

    def forward(self, args):
        output = self.main(args)
        if self.flatten is not None:
            output = self.flatten(output)

        return output

    def condense_in_model(self, model: torch.nn.Module):
        # Plan: delete all but the first layer and replace the first layer with self.main

        test = self.replacing.modules()
        first = test.__next__()  # This is just the container of the modules
        first = test.__next__()  # This is the actual first module

        for name, module in model.named_modules():
            # because we need to track down the actual path of the module, this method is only applied to named modules
            if module in self.replacing.modules():
                # We split the path on "." to get each individual attribute name
                name_path = name.split(".")

                # Current_look is/will be the container module for the module we are replacing
                current_look = model
                for i in name_path[:-1]:
                    if isinstance(current_look, Nothing_Module):
                        # This should not happen but if the container has already been replaced and there is still more of the path,
                        # Then end the replacemnt because the rest has been replaced already.
                        break
                    # Traversing the model to find the container module
                    current_look = current_look.__getattr__(i)

                if module is first:
                    current_look.__setattr__(name_path[-1], self.main)
                elif module is self.flatten:
                    pass
                elif isinstance(current_look, Nothing_Module):
                    pass
                else:
                    # Replace the modules in the predicessor with a module that does nothing
                    current_look.__setattr__(name_path[-1], Nothing_Module(current_look.__getattr__(name_path[-1])))

        # print(model.state_dict())
        self.rm1.remove()
        self.rm2.remove()

    def update_p(self, t):
        self.p = min(1, self.b + self.k * t)
