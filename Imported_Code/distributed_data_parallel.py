# Code from: https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
# I may try to modify it?
import os

# Block 1
import torch.distributed as dist
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Block 2
from torch.utils.data.distributed import DistributedSampler
def prepare(rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    dataset = Your_Dataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

# def distributed_data_parallel(config, dataloader, model):
#     global imported_distributed_stuff
#     if config("Dataparallel") == 1:
#         if not imported_distributed_stuff:
#             import os
#             import torch.distributed as dist
#             from torch.utils.data.distributed import DistributedSampler
#             from torch.utils.data import DataLoader
#             from torch.nn.parallel import DistributedDataParallel as DDP
#             imported_distributed_stuff = True

#         # https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
#         os.environ['MASTER_ADDR'] = 'localhost'
#         os.environ['MASTER_PORT'] = '12355'
#         dist.init_process_group("nccl", rank=rank), world_size=world_size)

#         dataset = dataloader.dataset
#         distribution_of_data = DistributedSampler(dataset, num_replicas=config("NumberOfWorkers"))

#         dataloader = DataLoader(dataset=dataset, batch_size=config("BatchSize"), collate_fn=datareader.collate_fn_, sampler=distribution_of_data)

#         model.to(config("NumberOfWorkers"))