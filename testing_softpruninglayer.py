from src import *

# args = standard_run()


module = torch.nn.Linear(7, 13, dtype=torch.float, bias=False)

test_data = torch.tensor(range(7 * 12), dtype=torch.float).reshape(12, -1)

print(test_data)

# module.weight.data[::3] = 0
# module.weight.data[1::3] = 0.5
# module.weight.data[2::3] = 1

basic_output = module(test_data)

print(basic_output)

remove = modelstruct.PreSoftPruningLayer(module)

remove.para.data = torch.tensor([x % 2 for x in range(7)], dtype=torch.float)

with_active_prelayer = module(test_data)

print(with_active_prelayer)

remove.remove()

with_inactive_prelayer = module(test_data)

for x, y in zip(with_active_prelayer, with_inactive_prelayer):
    for a, b in zip(x, y):
        if a-b != 0:
            print(a-b)

print("Done pt 1")

remove = modelstruct.PostSoftPruningLayer(module)

remove.para.data = torch.tensor([x % 2 for x in range(13)], dtype=torch.float)

with_active_postlayer = module(test_data)

print(with_active_postlayer)

remove.remove()

with_inactive_postlayer = module(test_data)

for x, y in zip(with_active_postlayer, with_inactive_postlayer):
    for a, b in zip(x, y):
        if a-b != 0:
            print(a-b)

print("Done")