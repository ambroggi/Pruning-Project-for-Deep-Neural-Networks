import torch


def collect_module_is(model: torch.nn.Module, paramNumbers: list, batch: torch.Tensor):
    i = 0
    hooks = []
    removers = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            if i in paramNumbers:
                # Prepare to catch the data for the module
                hooks.append(forward_hook())
                removers.append(module.register_forward_hook(hooks[-1]))
            i += 1

    model(batch)

    for r in removers:
        r.remove()

    return hooks


def run_one_channel_module(module: torch.nn.Module, dat: torch.Tensor):
    state_dict = {a: b.clone() for a, b in module.state_dict().items()}
    lst = []
    for x in range(len(state_dict["weight"])):
        state_dict_clone = {a: b.clone() for a, b in state_dict.items()}
        state_dict_clone["bias"] = torch.zeros_like(state_dict_clone["bias"])
        state_dict_clone["weight"] = torch.zeros_like(state_dict_clone["weight"])
        state_dict_clone["weight"][x] = state_dict["weight"][x]

        module.load_state_dict(state_dict_clone)
        out = module(dat)
        while out.ndim > 1:
            out = torch.sum(out, dim=1)
        lst.append(out)

    module.load_state_dict(state_dict)
    return lst


def remove_layers(model: torch.nn.Module, parameter_to_reduce: int, keepint_tensor, length_of_single_channel=1):

    state_dict = {}
    idx = -1
    for names, params in model.named_parameters():
        if "weight" in names:
            idx += 1
            if idx == parameter_to_reduce:
                state_dict = state_dict | {f"{names.split('.')[0]}.weight": params.data[keepint_tensor[:: length_of_single_channel]]}
            if idx == parameter_to_reduce + 1:

                if len(keepint_tensor) != len(params.data[0]):
                    multiplier = len(params.data[0])//len(keepint_tensor)
                    t2 = torch.zeros(len(params.data[0]), dtype=torch.bool)
                    for i in range(len(keepint_tensor)):
                        t2[i*multiplier:(i+1)*multiplier] = keepint_tensor[i]
                    state_dict = state_dict | {f"{names.split('.')[0]}.weight": params.data[:, t2]}

                else:
                    state_dict = state_dict | {f"{names.split('.')[0]}.weight": params.data[:, keepint_tensor]}

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
