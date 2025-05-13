#torchrun --nproc_per_node=2 tensor_parallel_example_toy.py 

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
logger = get_logger()

# ---- GPU check ------------
_min_gpu_count = 2

if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} GPUs to run this example. Exiting.")
    sys.exit()
# ---------------------------

from torch.distributed._tensor.device_mesh import init_device_mesh

"""
This script tests Tensor Parallelism (TP) on a toy model in a Megatron-LM SPMD style.
We demonstrate an end-to-end workflow from forward pass, backward pass, to optimization.

The model consists of multiple `nn.Linear` layers with `nn.ReLU` activations in between.
We alternate between column-wise and row-wise parallelism for the linear layers.
"""

class ToyModel(nn.Module):
    """MLP-based model with parameterized layers."""

    def __init__(self, k):
        super(ToyModel, self).__init__()
        assert k % 2 == 0, "The number of layers k must be even."
        self.layers = nn.ModuleList()
        input_size = 10  # Initial input size
        hidden_size = 32  # Hidden layer size

        # Create k Linear layers with ReLU activations in between
        for i in range(k):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))

            # Add ReLU activation between layers, except after the last layer
            if i < k - 1:
                self.layers.append(nn.ReLU())

        # Output projection layer
        self.out_proj = nn.Linear(hidden_size, 5)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            with record_function(f"layer_{idx}"):
                x = layer(x)
        x = self.out_proj(x)
        return x

"""
Main body of the demo for a parameterized version of tensor parallelism using
PyTorch native APIs.
"""
logger = get_logger()

# Get the world size and initialize the device mesh
_world_size = int(os.environ["WORLD_SIZE"])

device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()

print(f"Starting PyTorch TP example on rank {_rank}.")
assert (
    _world_size % 2 == 0
), f"TP examples require an even number of GPUs, but got {_world_size} GPUs"

rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")

# Parameter k: number of layers (must be even)
k = 4  # You can change this value to any even number
assert k % 2 == 0, "Parameter k must be even to alternate parallelism styles."

# Create the model and move it to GPU
tp_model = ToyModel(k=k).to("cuda")

# Create an optimizer for the parallelized module
lr = 0.25
optimizer = torch.optim.AdamW(tp_model.parameters(), lr=lr, foreach=True)

# Custom parallelization plan for the model
parallelize_plan = {}

# Iterate over the layers and assign parallelization strategies
for idx, layer in enumerate(tp_model.layers):
    if isinstance(layer, nn.Linear):
        # Calculate the layer number among Linear layers
        linear_layer_number = idx // 2  # Every second layer is a Linear layer
        if linear_layer_number % 2 == 0:
            # Even-numbered Linear layers use ColwiseParallel
            parallelize_plan[f"layers.{idx}"] = ColwiseParallel()
        else:
            # Odd-numbered Linear layers use RowwiseParallel
            parallelize_plan[f"layers.{idx}"] = RowwiseParallel()

# Parallelize the output projection layer
# Alternate based on k to ensure matching parallelism style
if (k // 2) % 2 == 0:
    # If k/2 is even, use ColwiseParallel
    parallelize_plan["out_proj"] = ColwiseParallel()
else:
    # If k/2 is odd, use RowwiseParallel
    parallelize_plan["out_proj"] = RowwiseParallel()

# Apply the parallelization plan to the model
tp_model = parallelize_module(
    module=tp_model,
    device_mesh=device_mesh,
    parallelize_plan=parallelize_plan,
)

print(tp_model)

# Run a forward pass and export profiling trace
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        inp = torch.rand(20, 10, device="cuda")
        output = tp_model(inp)

prof.export_chrome_trace(f"trace{_rank}.json")
print("Trace exported")

    
