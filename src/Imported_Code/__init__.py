import sys
import os
from typing import TYPE_CHECKING

import torch
import torch.utils.data

if TYPE_CHECKING:
    from src.modelstruct import BaseDetectionModel

from .helperFunctions import (ConfigCompatibilityWrapper, forward_hook,
                              get_layer_by_state, remove_layers,
                              set_layer_by_state)

try:
    added_path = os.path.join("src", "Imported_Code", "admm_joint_pruning", "joint_pruning")
    sys.path.append(added_path)
    from .admm_joint_pruning.joint_pruning.main import (apply_filter,
                                                        apply_l1_prune,
                                                        apply_prune,
                                                        prune_admm)
    sys.path.remove(added_path)

    from .ADMM_extra import add_admm_v_layers, remove_admm_v_layers
    ADMM_EXISTS = True
except ModuleNotFoundError:
    ADMM_EXISTS = False

from .BERT_Theseus_From_Paper import Theseus_Replacement
from .DAIS_From_Paper import DAIS_fit, add_alpha
from .ThiNet_extra import run_thinet_on_layer
from .ThiNet_From_Paper import thinet_pruning

try:
    from .Task_Oriented_Feature_Distillation_implementation import (
        TOFD_name_main, task_oriented_feature_wrapper)
    TOFD_EXISTS = True
except ModuleNotFoundError:
    TOFD_EXISTS = False

if torch.utils.data.get_worker_info() is None:
    print(f"Imported code __init__ file loaded as {__name__}.")
else:
    pass


if TYPE_CHECKING:
    # This is purely so that the linter does not yell at me saying "module imported but not used"
    BaseDetectionModel
    remove_layers, get_layer_by_state, set_layer_by_state, forward_hook, ConfigCompatibilityWrapper
    prune_admm, apply_filter, apply_prune, apply_l1_prune
    add_admm_v_layers, remove_admm_v_layers
    thinet_pruning
    run_thinet_on_layer
    Theseus_Replacement
    add_alpha, DAIS_fit
    task_oriented_feature_wrapper, TOFD_name_main
