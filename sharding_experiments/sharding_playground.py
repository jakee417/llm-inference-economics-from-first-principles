import os
import sys
import torch
import torch.nn as nn

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

from log_utils import rank_log, get_logger, verify_min_gpu_count
from torch.profiler import profile, record_function, ProfilerActivity
from torch.distributed.device_mesh import init_device_mesh

logger = get_logger()


# Create a 3x6 matrix filled with ones and get the lower triangular part
matrix = torch.tril(torch.ones((3, 6)))
print(matrix)


# Get the world size and initialize the device mesh
_world_size = int(os.environ["WORLD_SIZE"])

device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()

print(f"Starting PyTorch TP example on rank {_rank}.")
assert (
    _world_size % 2 == 0
), f"TP examples require an even number of GPUs, but got {_world_size} GPUs"

rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")
