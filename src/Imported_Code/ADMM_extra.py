# This is just some extra adaptation code for ADMM that I had made that does not fit elsewhere so I just made it its own file.

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.modelstruct import BaseDetectionModel

from ..extramodules import PostMutablePruningLayer


def add_admm_v_layers(model: "BaseDetectionModel"):
    count = 1
    for module in model.get_important_modules():
        print(module)
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv1d):
            model.pruning_layers.append(PostMutablePruningLayer(module))
            model.pruning_layers[-1].para.to(model.cfg("Device"))
            # model.pruning_layers[-1].para.data = torch.rand_like(model.pruning_layers[-1].para.data)
            model.register_parameter(f"v{count}", model.pruning_layers[-1].para)
            count += 1


def remove_admm_v_layers(model: "BaseDetectionModel"):
    count = 1
    # test = [x.para.data for x in model.pruning_layers]
    while len(model.pruning_layers) > 0:
        model.__setattr__(f"v{count}", None)
        pruning_layer = model.pruning_layers.pop(0)
        pruning_layer.remove()
        count += 1


if __name__ == "__main__":
    # Test that the method of using v layers does not change the gradient from the original version

    # Testing input tensor
    t = torch.Tensor([
            [1, 2, 3, 4, 5, 6],
            [2, 2, 3, 4, 5, 6],
            [3, 2, 3, 4, 5, 6],
            [4, 2, 3, 4, 5, 6],
            [5, 2, 3, 4, 5, 6],
            [6, 2, 3, 4, 5, 6]
        ]
    )

    # Testing V values
    v = torch.Tensor([0, 1, 0, 1, 0, 1])

    # Check that the diagonalization works and what it does.
    print(torch.diag(v).mm(t))

    # Create a model with the added v layers
    module = torch.nn.Linear(6, 6, bias=True)
    module.register_parameter("v", torch.nn.Parameter(v))
    loss_funct = torch.nn.MSELoss()

    # Save the starting state
    state = {x: y.clone() for x, y in module.state_dict().items()}

    # Account for randomness
    torch.random.manual_seed(1)
    # Account for internal state of optimizer
    opt = torch.optim.SGD(module.parameters(), lr=1, momentum=0)

    # Apply method 1
    opt = torch.optim.SGD(module.parameters(), lr=1, momentum=0)
    out = torch.nn.functional.linear(t, torch.diag(v).mm(module.weight), module.bias)
    loss = loss_funct(out, t)
    loss.backward()
    opt.step()
    module.zero_grad()
    print(module.state_dict())

    # Reload module
    module.load_state_dict(state)
    # Account for randomness
    torch.random.manual_seed(1)
    # Account for internal state of optimizer
    opt = torch.optim.SGD(module.parameters(), lr=1, momentum=0)

    # Attempt method 2 (the method we use)
    out = module(t)*v[None, :]
    loss = loss_funct(out, t)
    loss.backward()
    opt.step()
    module.zero_grad()
    print(module.state_dict())  # This should be the same as the last print.

    # Sanity check method 1 again (We originally had differences between the two methods but that was due to non-zeroed gradients)
    module.load_state_dict(state)
    torch.random.manual_seed(1)
    opt = torch.optim.SGD(module.parameters(), lr=1, momentum=0)
    # Method 1 again
    opt = torch.optim.SGD(module.parameters(), lr=1, momentum=0)
    out = torch.nn.functional.linear(t, torch.diag(v).mm(module.weight), module.bias)
    loss = loss_funct(out, t)
    loss.backward()
    opt.step()
    module.zero_grad()
    print(module.state_dict())
