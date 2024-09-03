# This is just some extra adaptation code for ADMM that I had made that does not fit elsewhere so I just made it its own file.

import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.modelstruct import BaseDetectionModel
from ..extramodules import PostMutablePruningLayer


def add_addm_v_layers(model: "BaseDetectionModel"):
    count = 1
    for module in model.get_important_modules():
        print(module)
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv1d):
            model.pruning_layers.append(PostMutablePruningLayer(module))
            # model.pruning_layers[-1].para.data = torch.rand_like(model.pruning_layers[-1].para.data)
            model.register_parameter(f"v{count}", model.pruning_layers[-1].para)
            count += 1


def remove_addm_v_layers(model: "BaseDetectionModel"):
    count = 1
    # test = [x.para.data for x in model.pruning_layers]
    while len(model.pruning_layers) > 0:
        model.__setattr__(f"v{count}", None)
        pruning_layer = model.pruning_layers.pop(0)
        # Only keeps the top magnitude weights as described in section "4.4. Network retraining"
        top = torch.topk(pruning_layer.para.data, len(pruning_layer.para.data) - model.cfg("LayerPruneTargets")[count-1], largest=False).indices
        pruning_layer.para.data.view(-1)[top] = 0
        pruning_layer.remove()
        count += 1
