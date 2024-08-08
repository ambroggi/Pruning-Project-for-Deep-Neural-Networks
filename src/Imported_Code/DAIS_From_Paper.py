# This is an implementation of DAIS from the paper: DAIS: Automatic Channel Pruning via Differentiable Annealing Indicator Search (https://arxiv.org/pdf/2011.02166)
# Implementation made by Alexandre Broggi 2024, I hope I am not making any big mistakes
import torch
from .. import modelstruct


class add_alpha(modelstruct.PreMutablePruningLayer):

    def __init__(self, module: torch.nn.Module, target_percent: float):
        super().__init__(self, module)
        self.T_value = 1
        self.T_anneal = lambda x: 0.1/x
        self.epoch = 0
        self.pl = torch.prod(module.weight.shape)  # In equation 9
        self.target_percent = target_percent

    def __call__(self, module: torch.nn.Module, args: list[torch.Tensor], output: torch.Tensor):
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

    def lasso_reg(self):
        # Note this is equation 8 of the paper except for the largest sum,
        # which needs to be applied on the results of this object
        return torch.sum(torch.norm((self.HT()), p=1))

    def HT(self):
        return torch.sigmoid(self.para/self.T_value)

    @staticmethod
    def regulizer_loss(lst):
        lasso = torch.sum(torch.cat([x.lasso_reg() for x in lst]))

        E_flops = sum([(a.pl * sum(a.HT()) * sum(b.HT())) for a, b in zip(lst, lst[1:])])
        F = sum([(a.pl * (len(a.HT()) * a.target_percent) * (len(b.HT()) * b.target_percent)) for a, b in zip(lst, lst[1:])])

        if E_flops/F > 1:
            flops = torch.log(E_flops)
        elif E_flops/F < 0.999999:
            flops = -torch.log(E_flops)
        else:
            flops = torch.tensor(0)

        # We are not using residual blocks so we don't have a place for equation 11

        return lasso + flops
    # def e_flops(self):
    #     # This is equation 9, again, without the last sum

    def callback_fn(self):
        return lambda results: self.anneal(results["epoch"])
