import sys
import torch
import torch.utils.data
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.modelstruct import BaseDetectionModel
    BaseDetectionModel


# import os

# for subpackage in glob.glob("src/Imported_Code"):
#     print(f"Adding \'{subpackage}\' to system path")
#     sys.path.append(subpackage)

from .helperFunctions import remove_layers, get_layer_by_state, set_layer_by_state, forward_hook, ConfigCompatabilityWrapper
remove_layers, get_layer_by_state, set_layer_by_state, forward_hook, ConfigCompatabilityWrapper

sys.path.append("src/Imported_Code/admm_joint_pruning/joint_pruning")
from .admm_joint_pruning.joint_pruning.main import prune_admm, apply_filter, apply_prune, apply_l1_prune
prune_admm, apply_filter, apply_prune, apply_l1_prune
sys.path.remove("src/Imported_Code/admm_joint_pruning/joint_pruning")

from .ADMM_extra import add_addm_v_layers, remove_addm_v_layers
add_addm_v_layers, remove_addm_v_layers

from .ThiNet_From_Paper import thinet_pruning
thinet_pruning

from .ThiNet_extra import run_thinet_on_layer
run_thinet_on_layer

from .BERT_Theseus_From_Paper import Theseus_Replacement
Theseus_Replacement

from .DAIS_From_Paper import add_alpha, DAIS_fit
add_alpha, DAIS_fit

from .Task_Oriented_Feature_Distillation_implementation import task_oriented_feature_wrapper, TOFD_name_main
task_oriented_feature_wrapper, TOFD_name_main


if torch.utils.data.get_worker_info() is None:
    print(f"Imported code __init__ file loaded as {__name__}.")
else:
    pass
