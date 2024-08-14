import torch


class PreMutablePruningLayer():
    def __init__(self, module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            self.para = torch.nn.Parameter(torch.ones(module.in_features))
        elif isinstance(module, torch.nn.Conv1d):
            self.para = torch.nn.Parameter(torch.ones(module.in_channels))
        else:
            print(f"Soft Pruning for Module type {module._get_name()} not implemented yet")
        self.module = module
        module.register_parameter(f"v_{module._get_name()}", self.para)
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
        self.module.__setattr__(f"v_{self.module._get_name()}", None)
        del self.para


class PostMutablePruningLayer():
    def __init__(self, module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            self.para = torch.nn.Parameter(torch.ones(module.out_features))
        elif isinstance(module, torch.nn.Conv1d):
            self.para = torch.nn.Parameter(torch.ones(module.out_channels))
        else:
            print(f"Soft Pruning for Module type {module._get_name()} not implemented yet")
        self.module = module
        module.register_parameter(f"v_{module._get_name()}", self.para)
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
        self.module.__setattr__(f"v_{self.module._get_name()}", None)
        del self.para


class Nothing_Module(torch.nn.Module):
    def __init__(self, old: torch.nn.Module):
        super().__init__()
        self.old = [old]  # Making it a list so torch cannot find it

    def forward(self, args, **kwargs):
        return args

    def set_extra_state(self, state):
        pass

    def get_extra_state(self) -> str:
        return "NONE"
