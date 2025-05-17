import torch


class PreMutablePruningLayer():
    def __init__(self, module: torch.nn.Module, register_parameter=True):
        if isinstance(module, torch.nn.Linear):
            self.para = torch.nn.Parameter(torch.ones(module.in_features, device=module.weight.device))
        elif isinstance(module, torch.nn.Conv1d):
            self.para = torch.nn.Parameter(torch.ones(module.in_channels, device=module.weight.device))
        else:
            print(f"Soft Pruning for Module type {module._get_name()} not implemented yet")
        self.module = module

        if register_parameter:
            self.parameter = True
            module.register_parameter(f"v_{module._get_name()}", self.para)
        else:
            self.parameter = False
        self.remove_hook = module.register_forward_pre_hook(self)

    def __call__(self, module: torch.nn.Module, args: tuple[torch.Tensor]) -> torch.Tensor:
        if isinstance(module, torch.nn.Linear):
            return args[0] * self.para[None, :]
        elif isinstance(module, torch.nn.Conv1d):
            return args[0] * self.para[None, :, None]
        else:
            print("Soft Pruning Layer Failed")
            return args[0]

    def remove(self, update_weights: bool = True):
        self.remove_hook.remove()
        if update_weights:
            self.module.__getattr__("weight").data *= self.para.data

        if self.parameter:
            self.module.__setattr__(f"v_{self.module._get_name()}", None)
        del self.para


class PostMutablePruningLayer():
    def __init__(self, module: torch.nn.Module, register_parameter=True):
        if isinstance(module, torch.nn.Linear):
            self.para = torch.nn.Parameter(torch.ones(module.out_features, device=module.weight.device))
        elif isinstance(module, torch.nn.Conv1d):
            self.para = torch.nn.Parameter(torch.ones(module.out_channels, device=module.weight.device))
        else:
            print(f"Soft Pruning for Module type {module._get_name()} not implemented yet")
        self.module = module

        if register_parameter:
            self.parameter = True
            module.register_parameter(f"v_{module._get_name()}", self.para)
        else:
            self.parameter = False
        self.remove_hook = module.register_forward_hook(self)

    def __call__(self, module: torch.nn.Module, args: list[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
        if isinstance(module, torch.nn.Linear):
            return output * self.para[None, :]
        elif isinstance(module, torch.nn.Conv1d):
            return output * self.para[None, :, None]
        else:
            print("Soft Pruning Layer Failed")
            return output

    def remove(self, update_weights: bool = True):
        self.remove_hook.remove()
        if update_weights:
            w: torch.nn.Parameter = self.module.__getattr__("weight")
            self.module.__getattr__("weight").permute(*torch.arange(w.ndim - 1, -1, -1)).data *= self.para.data
            self.module.__getattr__("bias").permute(*torch.arange(self.module.__getattr__("bias").ndim - 1, -1, -1)).data *= self.para.data

        if self.parameter:
            self.module.__setattr__(f"v_{self.module._get_name()}", None)
        del self.para


class Nothing_Module(torch.nn.Module):
    def __init__(self, old: torch.nn.Module):
        """
        This is just a module that does nothing so that it can store the place where an actual module used to be.

        Args:
            old (torch.nn.Module): The module that this is replacing. It is stored as self.old[0] so that it cannot be found by torch autograd
        """
        super().__init__()
        self.old = [old]  # Making it a list so torch cannot find it. (Torch cannot find things in default lists, that is why the module list exists)

    def forward(self, args, **kwargs):
        return args

    def set_extra_state(self, state):
        pass

    def get_extra_state(self) -> str:
        return "NONE"


class Get_Average_Hook():
    def __init__(self):
        self.dict = {}
        self.average = None
        self.count = 0

    def __call__(self, module: torch.nn.Module, _, new_values: torch.Tensor):
        with torch.no_grad():
            if module in self.dict.keys():
                count, average = self.dict[module]
                new_count = count + len(new_values)
                average = ((average*count) + new_values.sum(dim=0))/new_count
                count = new_count
                self.dict[module] = count, average
            else:
                self.dict[module] = len(new_values), new_values.sum(dim=0)
