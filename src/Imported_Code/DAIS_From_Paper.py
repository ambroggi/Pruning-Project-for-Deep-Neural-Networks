# This is an implementation of DAIS from the paper: DAIS: Automatic Channel Pruning via Differentiable Annealing Indicator Search (https://arxiv.org/pdf/2011.02166)
# Implementation made by Alexandre Broggi 2024, I hope I am not making any big mistakes
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.dataloader
from ..extramodules import PostMutablePruningLayer
from ..modelstruct import BaseDetectionModel


class add_alpha(PostMutablePruningLayer):

    def __init__(self, module: torch.nn.Module, target_percent: float, max_epochs: int):
        super().__init__(module)
        self.para.data *= torch.rand_like(self.para.data)
        self.T_value = 1
        self.T_anneal = lambda x: 1/(1+(49 * x/max_epochs))
        self.epoch = 0
        self.pl = torch.prod(torch.tensor([x for x in module.weight.shape]))  # In equation 9 (or at least I think that is how it is calculated?)
        self.target_percent = target_percent

    def __call__(self, module: torch.nn.Module, args: list[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
        # I_relaxed is described in equation 7
        I_relaxed = self.HT()

        # Apply to all layer outputs
        if isinstance(module, torch.nn.Linear):
            return output * I_relaxed[None, :]
        elif isinstance(module, torch.nn.Conv1d):
            return output * I_relaxed[None, :, None]
        else:
            print("Soft Alpha Layer Failed")
            return output

    def anneal(self, epoch=None, save_epoch=False):
        if epoch is not None:
            self.T_value = self.T_anneal(epoch)
            if save_epoch:
                self.epoch = epoch
        else:
            self.epoch += 1
            self.T_value = self.T_anneal(self.epoch)

    def lasso_reg(self) -> torch.Tensor:
        # Note this is equation 8 of the paper except for the largest sum,
        # which needs to be applied on the results of this object
        return torch.sum(torch.norm((self.HT()), p=1))

    def HT(self) -> torch.Tensor:
        return torch.sigmoid(self.para/self.T_value)

    # def e_flops(self):
    #     # This is equation 9, again, without the last sum

    def callback_fn(self, results) -> None:
        return self.anneal(results["epoch"] + 1)

    def remove(self):
        self.para = self.HT().greater(0.5)
        super().remove()


def regulizer_loss(lst: list[add_alpha]):
    # Equation 8
    lasso = torch.sum(torch.stack([x.lasso_reg() for x in lst]))

    # Equation 9... I think? Pl is not defined well
    E_flops = sum([(b.pl * sum(a.HT()) * sum(b.HT())) for a, b in zip(lst, lst[1:])])
    F = sum([(b.pl * (len(a.HT()) * a.target_percent) * (len(b.HT()) * b.target_percent)) for a, b in zip(lst, lst[1:])])

    # Equation 10, or it should be, F is not well defined as far as I can tell.
    if E_flops/F > 1:
        flops = torch.log(E_flops)
    elif E_flops/F < 0.9999:
        flops = -torch.log(E_flops)
    else:
        flops = torch.tensor(0)

    # We are not using residual blocks so we don't have a place for equation 11

    return lasso + flops


def DAIS_fit(model: BaseDetectionModel, alpha_hooks: list[add_alpha], epochs: int = 0, dataloader: DataLoader | None = None, keep_callbacks=False):
    # This is an implementation of the DAIS regularizer and DARTS (https://arxiv.org/pdf/1806.09055) search method.
    # DARTS is used because that is the method that DAIS says that their method is based on.
    # The exact DAIS method is supposedly in a different file, that I cannot find. "more details provided in the reference paper" - DAIS

    # Find the alpha parameters
    alp = [a.para for a in alpha_hooks]

    assert True in [alp[0] is x for x in model.parameters()]

    # Fit set-up
    if dataloader is None:
        if model.dataloader is None:
            raise TypeError("No dataset selected for Automatic Vulnerability Detection training")
        dl = model.dataloader
    else:
        dl = dataloader

    # Build the training and 'validation' datasets
    weight_dl, alpha_dl = torch.utils.data.random_split(dl.dataset, [0.7, 0.3])
    weight_dl = torch.utils.data.DataLoader(weight_dl, **(dl.base.load_kwargs if hasattr(dl, "base") else {}))
    alpha_dl = torch.utils.data.DataLoader(alpha_dl, **(dl.base.load_kwargs if hasattr(dl, "base") else {}))

    if model.optimizer is None:
        # This is really silly but it appears that parameters cannot be removed by .remove() if they are of different shapes?
        pram = [a for a in model.parameters() if True not in [a is b for b in alp]]
        model.optimizer = model.cfg("Optimizer")(pram, lr=model.cfg("LearningRate"))
    primary_optimizer = model.optimizer
    secondary_optimizer = model.cfg("Optimizer")(model.parameters(), lr=model.cfg("LearningRate"))

    if model.loss_fn is None:
        model.loss_fn = model.cfg("LossFunction")()

    model = model.to(model.cfg("Device"))

    frozen_bad = [x for x in model.frozen.keys() if (x not in model.state_dict().keys()) or (model.frozen[x].shape != model.state_dict()[x].shape)]
    for incorrect_frozen in frozen_bad:
        model.frozen.pop(incorrect_frozen)

    progres_bar = model.get_progress_bar(epochs)

    for e in progres_bar if progres_bar is not None else range(epochs):
        # Find speculative weights (This is training the model weights). DARTS Algorithm 1, step 1, estimate W*
        model.loss_additive_info = torch.zeros, (1, )
        model.optimizer = primary_optimizer
        non_speculative_weights = {x: y for x, y in model.state_dict().items() if "v" not in x}  # Because of addm all of the alpha weights are called "v_"
        epoch_results = model.run_single_epoch(weight_dl)
        e_results = {f"spec_{x[0]}": x[1] for x in epoch_results.items()}

        # Find new alpha values. DARTS Algorithm 1, step 1, update alpha
        model.loss_additive_info = regulizer_loss, (alpha_hooks, )
        model.optimizer = secondary_optimizer
        model.zero_grad()
        epoch_results = model.run_single_epoch(alpha_dl)
        model.zero_grad()
        e_results = e_results | {f"alph_{x[0]}": x[1] for x in epoch_results.items()}

        # Actually train the weights. DARTS Algorithm 1, step 1, update alpha
        model.load_state_dict(non_speculative_weights, strict=False)  # First reset w* back to w
        model.loss_additive_info = torch.zeros, (1, )
        model.optimizer = primary_optimizer
        epoch_results = model.run_single_epoch(weight_dl)
        e_results = e_results | epoch_results

        # I AM NOT CALLING A TRAINING DATASET "VALIDATION", it will be known as alpha_dl. This is the actual validation dataset
        if model.validation_dataloader is not None:
            val_epoch_results = {f"val_{x[0]}": x[1] for x in model.run_single_epoch(model.validation_dataloader).items()}
        else:
            val_epoch_results = {f"val_{x}": 0.0 for x in epoch_results.keys()}
        e_results = e_results | val_epoch_results

        e_results["epoch"] = e

        for call in model.epoch_callbacks:
            call(e_results)

    # Clear out the tqdm callback
    if progres_bar is not None:
        model.remove_progress_bar()

    if not keep_callbacks:
        model.epoch_callbacks = []

    return f'Ran model with {e_results["f1_score_macro"]*100:2.3f}% F1 on final epoch {e}'
