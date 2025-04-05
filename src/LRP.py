# Basing this off of https://git.tu-berlin.de/gmontavon/lrp-tutorial and http://www.zhyyyj.com/attachment/2021-6-17/cc5m64oxc0n79lslq5.pdf

if __name__ == "__main__":
    import pandas as pd
    import rdflib
    import torch.nn

    import __init__ as src


# from typing import Literal, TYPE_CHECKING
# These should stay the same for each model so I am just going to cache them instead of rebuilding.
global dataset, datasets
dataset = None
datasets = None


def get_model_and_datasets(csv_row: str | int = "0", csv_file: str = "results/BigModel(toOntology).csv") -> tuple["src.modelstruct.BaseDetectionModel", "src.getdata.ModifiedDataloader", list["src.getdata.ModifiedDataloader"]]:
    """
    Loads the model and the dataset from a specific csv file and row.

    Args:
        csv_row (str | int, optional): Row of the csv file to retrieve the model from. Defaults to "0".
        csv_file (str, optional): Csv file to read. Defaults to "results/BigModel(toOntology).csv".

    Returns:
        src.modelstruct.BaseDetectionModel: Model that has been loaded.
        src.getdata.ModifiedDataloader: Dataloader to read for general data.
        list:
            src.getdata.ModifiedDataloader: Dataloaders sorted by class.
    """
    print("Loading model and splitting dataset")
    config = src.cfg.ConfigObject()
    csv_row = str(csv_row)
    config("FromSaveLocation", f"csv {csv_row}" if csv_row.isdigit() else csv_row)
    config.readOnly.remove("ResultsPath")
    config.writeOnce.append("ResultsPath")
    config("ResultsPath", csv_file)
    config("PruningSelection", "Reset")
    load = src.standardLoad(existing_config=config, index=0, structure_only=False)
    global dataset, datasets
    if not dataset or not datasets:
        print("Building datasets from config")
        train, dataset = src.getdata.get_train_test(load["config"])
        dataset = dataset.base
        datasets = src.getdata.split_by_class(src.getdata.get_dataloader(load["config"], train), [x for x in range(dataset.number_of_classes)], individual=True, config=load["config"])
        print("Done building datasets")
    model = src.modelstruct.getModel(load["config"])
    model.cfg = load["config"]
    return model, dataset, datasets


def build_base_facts(csv_row: str | int = "0", csv_file: str = "results/BigModel(toOntology).csv", random_=False) -> tuple[str, "rdflib.Graph"]:
    # First set up the model
    model, dataset, datasets = get_model_and_datasets(csv_row, csv_file)

    # Create a hook to gather
    store_obj = storage_for_layer()
    hook_remover = torch.nn.modules.module.register_module_forward_hook(store_obj)

    # Get all of the modules we will want
    modules = [*model.get_important_modules()]

    # Create container for Relevance
    R: list[torch.Tensor] = [None]*(len(modules)+1)

    # Run the data through the model to collect.
    inp_ = []
    for X, y in datasets[0]:
        model(X)
        inp_.append(X)

    # Remove the hook from the module (Note: this method of collecting the values is not very fast.)
    hook_remover.remove()

    # Get all of the layer activations plus initial input
    A = [torch.cat(inp_, dim=0).detach()] + [torch.cat(store_obj.dict_[mod], dim=0).detach() for mod in modules]

    # It is unclear if this should be single-example or average, I interpret it as Average?
    A = [x.mean(dim=0).detach() for x in A]

    # Last relevance is just the final layer
    R[-1] = (A[-1]/A[-1].sum(dim=0))

    for l_idx in range(0, len(R)-1)[::-1]:
        A[l_idx].requires_grad = True
        A[l_idx].retain_grad()
        R[l_idx] = relprop(A[l_idx], modules[l_idx], R[l_idx+1])
        A[l_idx].grad = None
        model.zero_grad()

    # https://git.tu-berlin.de/gmontavon/lrp-tutorial
    # for l in range(1,len(R))[::-1]:

    #     w = rho(W[l],l)
    #     b = rho(B[l],l)

    #     z = incr(A[l].dot(w)+b,l)    # step 1
    #     s = R[l+1] / z               # step 2
    #     c = s.dot(w.T)               # step 3
    #     R[l] = A[l]*c                # step 4


# http://www.zhyyyj.com/attachment/2021-6-17/cc5m64oxc0n79lslq5.pdf
def relprop(a: "torch.Tensor", layer: "torch.nn.Module", R: "torch.Tensor"):
    # print(f"a: {a.sum()}")
    new_layer = rho(layer)
    z = epsilon + new_layer.forward(a)
    s = R/(z+1e-9)
    # print(f"{s.sum().item()=}\t{a.sum().item()=}")
    # print(f"zs - R: {(z*s.data - R).sum()}")
    # print(f"z/z: {(z/z.sum()).sum()}")
    (z*s.data).sum().backward()
    c = a.grad
    pass
    # c2 = torch.stack([(new_layer.weight[:, x]*s).sum() for x in range(len(a))])
    # print(f"c: {c.sum()}\t{(c - c2).sum().item()=}\tc/z: {c.sum()/(z.sum())}")
    R = a*c
    print(f"R: {R.sum()}")
    # return R.softmax(dim=0)
    return R


def rho(mod: "torch.Module"):
    if isinstance(mod, torch.nn.Linear):
        new_ = torch.nn.Linear(mod.in_features, mod.out_features)
        new_.weight.data = mod.weight.data + (abs(mod.weight.data) * 0.5)
        new_.bias.data = mod.bias.data + (abs(mod.bias.data) * 0.5)
        # new_.bias.data = torch.zeros_like(new_.bias.data)
        # return torch.nn.Sequential(new_)  # , torch.nn.ReLU()
        return new_


epsilon = 0.0


def select_best_rows(csv_file: str = "results/BigModel(toOntology).csv") -> list[int]:
    # Wanted to make this automatic but did not eventually do that.
    df = pd.read_csv("results/BigModel(toOntology).csv")
    # df = df[df["WeightPrunePercent"] == "[0.62, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 1.0]"]
    df = df[df["Notes"] == 0]
    return list(df.index)


class storage_for_layer():
    def __init__(self):
        self.dict_: dict["torch.nn.Module", list["torch.Tensor"]] = {}
        self.average = None
        self.count = 0

    def __call__(self, module: "torch.nn.Module", input_: tuple["torch.Tensor"], new_values: "torch.Tensor") -> "torch.Tensor":
        if module in self.dict_.keys():
            # self.dict_[module].append(input_[0])
            self.dict_[module].append(new_values)
        else:
            # self.dict_[module] = [input_[0]]
            self.dict_[module] = [new_values]


if __name__ == "__main__":
    if not False:
        for x in select_best_rows():
            build_base_facts(random_=False, csv_row=f"{x}")
        for x in [0, 1, 2]:
            build_base_facts(random_=True, csv_row=f"{x}")
            print(f"running for random {x}")
    else:
        print(select_best_rows())
