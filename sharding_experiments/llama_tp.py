import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn

from llama2_model import Transformer, ModelArgs

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)
from torch.profiler import profile, record_function, ProfilerActivity

def main():
    tp_size = 2  # Adjust this value based on the number of GPUs you want to use for TP

    # Initialize the distributed process group
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Starting PyTorch TP example on rank {rank}.")

    # Ensure the world size matches the TP size
    assert world_size == tp_size, f"World size {world_size} must be equal to TP size {tp_size}"

    # Set the CUDA device for this process
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Create a device mesh with only the tensor parallel dimension
    device_mesh = init_device_mesh(device.type, (tp_size,), mesh_dim_names=("tp",))

    # Create the model and move it to the correct GPU
    DIM = int(1024)
    simple_llama2_config = ModelArgs(dim=DIM, n_layers=4, n_heads=16, vocab_size=100)
    model = Transformer.from_model_args(simple_llama2_config).to(device)

    # Initialize model weights
    model.init_weights()

    # Parallelize the first embedding and the last linear output projection
    model = parallelize_module(
        model,
        device_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate()
            ),
        }
    )

    # Apply tensor parallelism to each transformer block
    for layer_id, transformer_block in enumerate(model.layers):
        layer_tp_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        }

        # Adjust attention module to use the local number of heads
        attn_layer = transformer_block.attention
        attn_layer.n_heads = attn_layer.n_heads // device_mesh.size()
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // device_mesh.size()

        # Apply the custom parallelization plan to the transformer block
        parallelize_module(
            module=transformer_block,
            device_mesh=device_mesh,
            parallelize_plan=layer_tp_plan
        )

    # Generate input and perform a single forward pass
    torch.manual_seed(0)
    inp = torch.randint(32000, (2048, 16), device=device)  # Input shape: [sequence_length, batch_size]
    _rank = device_mesh.get_rank()

    # warmups = 10
    # for _ in range(warmups):
    #     with torch.no_grad():
    #         output = model(inp)

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=False, record_shapes=True) as prof:
            with record_function('Model_Forward'):
                output = model(inp)
            print(f"Rank {rank} forward pass completed. Output shape: {output.shape}")
        
        import os
        try: 
            os.remove(f"llama_trace_4layers{_rank}_dim_{DIM}.json")
        except Exception:
            print("No file to delete")
        prof.export_chrome_trace(f"llama_trace_4layers{_rank}_dim_{DIM}.json")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()