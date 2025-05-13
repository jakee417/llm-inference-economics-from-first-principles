import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class ModelParams:
    """Parameters for a transformer-based LLM"""
    name: str = "LLama 3.3 70B"
    hidden_size: int = 8192
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    intermediate_size: int = 28672
    vocab_size: int = 128256
    dtype_bytes: int = 2  # bfloat16 = 2 bytes
    
    def __post_init__(self):
        # Calculate derived values
        self.head_size = self.hidden_size // self.num_attention_heads
        
        # Calculate total parameters
        self.total_params = self._calculate_total_params()
        
        # Calculate model size in bytes
        self.model_size_bytes = self.total_params * self.dtype_bytes
    
    def _calculate_total_params(self):
        """Calculate the total number of parameters in the model"""
        # Parameters per transformer block - simplified from the llama architecture
        block_params = (
            # Attention
            self.hidden_size * self.hidden_size +  # q projection
            self.hidden_size * self.hidden_size // 8 +  # k projection (8x smaller for LLama)
            self.hidden_size * self.hidden_size // 8 +  # v projection (8x smaller for LLama)
            self.hidden_size * self.hidden_size +  # o projection
            
            # MLP
            self.hidden_size * self.intermediate_size +  # gate projection
            self.hidden_size * self.intermediate_size +  # up projection
            self.intermediate_size * self.hidden_size +  # down projection
            
            # LayerNorms
            2 * self.hidden_size
        )
        
        # Embedding and head
        embedding_params = self.vocab_size * self.hidden_size
        head_params = self.hidden_size * self.vocab_size
        
        # Final layer norm
        final_norm_params = self.hidden_size
        
        # Total
        total = embedding_params + (self.num_hidden_layers * block_params) + final_norm_params + head_params
        
        return total


@dataclass
class HardwareParams:
    """Parameters for the hardware running the model"""
    name: str = "NVIDIA H100"
    tflops: float = 989  # Theoretical max FLOPS in TFLOPs for H100
    memory_bandwidth_GBs: float = 3350  # Memory bandwidth in GB/s for H100
    memory_size_GB: float = 80  # Memory size in GB
    num_gpus: int = 4  # Number of GPUs
    tensor_parallel: bool = True  # Whether to use tensor parallelism
    nvlink_bandwidth_GBs: float = 450  # NVLink bandwidth in GB/s
    
    def __post_init__(self):
        # Total memory across all GPUs
        self.total_memory_GB = self.memory_size_GB * self.num_gpus
        
        # Effective FLOPS and bandwidth when using multiple GPUs
        if self.tensor_parallel:
            # In tensor parallelism, we use all GPUs simultaneously
            self.effective_tflops = self.tflops * self.num_gpus
            self.effective_memory_bandwidth_GBs = self.memory_bandwidth_GBs * self.num_gpus
        else:
            # In pipeline parallelism, we use GPUs one after another (for throughput, not latency)
            self.effective_tflops = self.tflops
            self.effective_memory_bandwidth_GBs = self.memory_bandwidth_GBs


class TokenomicsModel:
    """Model for estimating LLM inference throughput"""
    
    def __init__(self, model_params, hardware_params):
        self.model = model_params
        self.hardware = hardware_params
        
    def calculate_prefill_flops(self, sequence_length):
        """Calculate FLOPs for the prefill phase"""
        S = sequence_length
        
        # Attention FLOPs (simplified from detailed calculations in the text)
        attention_flops = (
            # RMS Norm
            4 * S * self.model.hidden_size +
            
            # Query projection
            2 * S * self.model.hidden_size * self.model.hidden_size +
            
            # Key and value projections (LLama specific with reduced KV heads)
            0.5 * S * self.model.hidden_size * self.model.hidden_size +
            
            # Positional embedding (RoPE)
            6 * S * self.model.hidden_size +
            
            # Q @ K^T
            2 * S * S * self.model.hidden_size +
            
            # Softmax
            5 * S * S * self.model.num_attention_heads +
            
            # Attention output
            2 * S * S * self.model.hidden_size +
            
            # O-Projection
            2 * S * self.model.hidden_size * self.model.hidden_size
        )
        
        # MLP FLOPs
        mlp_flops = 21 * S * self.model.hidden_size * self.model.hidden_size
        
        # LM head FLOPs
        lm_head_flops = 2 * S * self.model.hidden_size * self.model.vocab_size
        
        # Total per layer and final
        per_layer_flops = attention_flops + mlp_flops
        total_flops = (per_layer_flops * self.model.num_hidden_layers) + lm_head_flops
        
        return total_flops
    
    def calculate_decode_flops(self, sequence_length):
        """Calculate FLOPs for a single token in the decode phase with KV cache"""
        # With KV caching, many operations are only performed for the new token,
        # but attention still needs to consider all previous tokens
        
        # Attention FLOPs with KV caching
        attention_flops = (
            # RMS Norm (for new token only)
            4 * self.model.hidden_size +
            
            # Query projection (for new token only)
            2 * self.model.hidden_size * self.model.hidden_size +
            
            # Key and value projections (for new token only, LLama specific)
            0.5 * self.model.hidden_size * self.model.hidden_size +
            
            # Positional embedding (for new token only)
            6 * self.model.hidden_size +
            
            # Q @ K^T (new token query against all cached keys)
            2 * sequence_length * self.model.hidden_size +
            
            # Softmax (across full sequence)
            5 * sequence_length * self.model.num_attention_heads +
            
            # Attention output (weighted sum with all values)
            2 * sequence_length * self.model.hidden_size +
            
            # O-Projection (for new token only)
            2 * self.model.hidden_size * self.model.hidden_size
        )
        
        # MLP FLOPs (for new token only)
        mlp_flops = 21 * self.model.hidden_size * self.model.hidden_size
        
        # LM head FLOPs (for new token only)
        lm_head_flops = 2 * self.model.hidden_size * self.model.vocab_size
        
        # Total per layer and final
        per_layer_flops = attention_flops + mlp_flops
        total_flops = (per_layer_flops * self.model.num_hidden_layers) + lm_head_flops
        
        return total_flops
    
    def calculate_kv_cache_size(self, sequence_length, batch_size=1):
        """Calculate the size of the KV cache in bytes"""
        # KV cache stores K and V values for each layer, head, and position
        kv_cache_bytes = (
            2 *  # K and V
            self.model.dtype_bytes *  # bytes per parameter
            self.model.num_hidden_layers *  # layers
            self.model.head_size *  # size per head
            self.model.num_key_value_heads *  # number of KV heads
            sequence_length *  # sequence length
            batch_size  # batch size
        )
        
        return kv_cache_bytes
    
    def prefill_time(self, sequence_length, batch_size=1):
        """Estimate the prefill time for processing the input prompt"""
        total_flops = self.calculate_prefill_flops(sequence_length) * batch_size
        total_tflops = total_flops / 1e12
        
        compute_time = total_tflops / self.hardware.effective_tflops
        
        comm_overhead = 0
        if self.hardware.tensor_parallel and self.hardware.num_gpus > 1:
            # Simple model for communication overhead
            # For each layer, we need to communicate activations between GPUs
            comm_size_bytes = batch_size * sequence_length * self.model.hidden_size * self.model.dtype_bytes
            comm_size_GB = comm_size_bytes / 1e9
            # Assume 2 communications per layer
            total_comm = 2 * self.model.num_hidden_layers * comm_size_GB
            # Time based on NVLink bandwidth
            comm_overhead = total_comm / self.hardware.nvlink_bandwidth_GBs
        
        return compute_time + comm_overhead
    
    def decode_time_breakdown(self, sequence_length, batch_size=1):
        """Estimate the compute and memory time for a single decode step"""
        # Calculate memory access time (loading model weights + KV cache)
        model_size_GB = self.model.model_size_bytes / 1e9
        kv_cache_size_GB = self.calculate_kv_cache_size(sequence_length, batch_size) / 1e9
        
        memory_load_time = (model_size_GB + kv_cache_size_GB) / self.hardware.effective_memory_bandwidth_GBs
        
        # Calculate compute time for one token
        decode_flops = self.calculate_decode_flops(sequence_length) * batch_size
        decode_tflops = decode_flops / 1e12
        compute_time = decode_tflops / self.hardware.effective_tflops
        
        # Communication overhead for tensor parallel (if applicable)
        comm_overhead = 0
        if self.hardware.tensor_parallel and self.hardware.num_gpus > 1:
            # For each token, we need to communicate smaller activations
            comm_size_bytes = batch_size * self.model.hidden_size * self.model.dtype_bytes
            comm_size_GB = comm_size_bytes / 1e9
            
            # Assume 2 communications per layer
            total_comm = 2 * self.model.num_hidden_layers * comm_size_GB
            
            # Time based on NVLink bandwidth
            comm_overhead = total_comm / self.hardware.nvlink_bandwidth_GBs
        
        # Determine the bottleneck
        bottleneck = "memory" if memory_load_time > compute_time else "compute"
        
        # Total time is dictated by the slower of compute or memory, plus communication overhead
        total_time = max(compute_time, memory_load_time) + comm_overhead
        
        return {
            "compute_time": compute_time,
            "memory_time": memory_load_time,
            "comm_overhead": comm_overhead,
            "bottleneck": bottleneck,
            "total_time": total_time
        }
    
    def decode_time(self, sequence_length, batch_size=1, generated_tokens=1):
        """Estimate the decode time for generating tokens after the prefill"""
        total_decode_time = 0
        
        # Current sequence length starts at prompt length
        current_sequence_length = sequence_length
        
        # For each token to generate
        for _ in range(generated_tokens):
            # Get the time breakdown for this token
            breakdown = self.decode_time_breakdown(current_sequence_length, batch_size)
            
            # Add to total time
            total_decode_time += breakdown["total_time"]
            
            # Increment sequence length for next token
            current_sequence_length += 1
        
        return total_decode_time
    
    def total_inference_time(self, input_tokens, output_tokens, batch_size=1):
        """Calculate the total time for processing a batch of sequences"""
        # Prefill phase time
        prefill = self.prefill_time(input_tokens, batch_size)
        
        # Decode phase time
        decode = self.decode_time(input_tokens, batch_size, output_tokens)
        
        return prefill + decode
    
    def calculate_total_throughput(self, input_tokens, output_tokens, batch_size=1):
        """Calculate throughput in tokens per second for the entire batch"""
        # Calculate total time
        time_seconds = self.total_inference_time(input_tokens, output_tokens, batch_size)
        
        # Calculate throughput (total tokens generated per second)
        total_tokens = batch_size * output_tokens
        throughput = total_tokens / time_seconds if time_seconds > 0 else 0
        
        return throughput
    
    def calculate_per_request_throughput(self, input_tokens, output_tokens, batch_size=1):
        """Calculate throughput in tokens per second per individual request"""
        # Calculate total throughput
        total_throughput = self.calculate_total_throughput(input_tokens, output_tokens, batch_size)
        
        # Per-request throughput is simply total throughput divided by batch size
        per_request_throughput = total_throughput / batch_size if batch_size > 0 else 0
        
        return per_request_throughput
    
    def analyze_bottlenecks(self, input_tokens, output_tokens, batch_sizes):
        """Analyze bottlenecks in decoding across different batch sizes"""
        compute_times = []
        memory_times = []
        total_times = []
        bottlenecks = []
        
        for batch_size in batch_sizes:
            # We'll analyze the bottleneck for the first token after the prompt
            breakdown = self.decode_time_breakdown(input_tokens, batch_size)
            
            compute_times.append(breakdown["compute_time"])
            memory_times.append(breakdown["memory_time"])
            total_times.append(breakdown["total_time"])
            bottlenecks.append(breakdown["bottleneck"])
        
        return {
            "compute_times": compute_times,
            "memory_times": memory_times,
            "total_times": total_times,
            "bottlenecks": bottlenecks
        }
    
    def plot_throughput_vs_batch_size(self, input_tokens=1024, output_tokens=128):
        """Plot total and per-request throughput against batch size with bottleneck analysis"""
        # Hardcoded batch sizes as requested
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        total_throughputs = []
        per_request_throughputs = []
        
        # Analyze bottlenecks
        bottleneck_analysis = self.analyze_bottlenecks(input_tokens, output_tokens, batch_sizes)
        compute_times = bottleneck_analysis["compute_times"]
        memory_times = bottleneck_analysis["memory_times"]
        bottlenecks = bottleneck_analysis["bottlenecks"]
        
        for batch_size in batch_sizes:
            total_throughput = self.calculate_total_throughput(input_tokens, output_tokens, batch_size)
            per_request_throughput = self.calculate_per_request_throughput(input_tokens, output_tokens, batch_size)
            
            total_throughputs.append(total_throughput)
            per_request_throughputs.append(per_request_throughput)
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Plot total throughput
        ax1.plot(batch_sizes, total_throughputs, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Total Throughput (tokens/second)')
        ax1.set_title(f'Theoretical Total Throughput vs Batch Size\n{self.model.name} on {self.hardware.num_gpus}x {self.hardware.name}\nInput Tokens: {input_tokens}, Output Tokens: {output_tokens}')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)  # Use log scale for x-axis
        ax1.set_xticks(batch_sizes)
        ax1.set_xticklabels(batch_sizes)
        
        # Add values above points in total throughput plot
        for i, v in enumerate(total_throughputs):
            ax1.text(batch_sizes[i], v + (max(total_throughputs) * 0.03), f"{v:.1f}", 
                    ha='center', va='bottom', fontweight='bold')
        
        # Plot per-request throughput
        ax2.plot(batch_sizes, per_request_throughputs, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Per-Request Throughput (tokens/second)')
        ax2.set_title(f'Theoretical Per-Request Throughput vs Batch Size\n{self.model.name} on {self.hardware.num_gpus}x {self.hardware.name}\nInput Tokens: {input_tokens}, Output Tokens: {output_tokens}')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)  # Use log scale for x-axis
        ax2.set_xticks(batch_sizes)
        ax2.set_xticklabels(batch_sizes)
        
        # Add values above points in per-request throughput plot
        for i, v in enumerate(per_request_throughputs):
            ax2.text(batch_sizes[i], v + (max(per_request_throughputs) * 0.03), f"{v:.1f}", 
                    ha='center', va='bottom', fontweight='bold')
        
        # Create a figure for bottleneck analysis
        fig2, ax3 = plt.subplots(figsize=(12, 7))
        
        # Plot compute and memory times
        ax3.plot(batch_sizes, compute_times, 'b-', linewidth=2, label='Compute Time')
        ax3.plot(batch_sizes, memory_times, 'r-', linewidth=2, label='Memory Time')
        
        # Add markers for bottlenecks
        for i, bottleneck in enumerate(bottlenecks):
            marker_color = 'blue' if bottleneck == 'compute' else 'red'
            marker_style = 'o' if bottleneck == 'compute' else 's'
            ax3.plot(batch_sizes[i], max(compute_times[i], memory_times[i]), 
                    marker=marker_style, markersize=10, color=marker_color)
        
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title(f'Compute vs Memory Bottleneck Analysis\n{self.model.name} on {self.hardware.num_gpus}x {self.hardware.name}\nInput Tokens: {input_tokens}, Output Tokens: {output_tokens}')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)  # Use log scale for x-axis
        ax3.set_xticks(batch_sizes)
        ax3.set_xticklabels(batch_sizes)
        ax3.legend()
        
        # Find crossover point
        for i in range(len(batch_sizes) - 1):
            if bottlenecks[i] != bottlenecks[i+1]:
                crossover_point = batch_sizes[i+1]
                ax3.axvline(x=crossover_point, color='green', linestyle='--', 
                        label=f'Bottleneck Shift at Batch Size {crossover_point}')
                ax3.legend()
                break
        
        # Create a data table with the values and bottlenecks
        data = []
        for i, bs in enumerate(batch_sizes):
            data.append((
                bs, 
                round(total_throughputs[i], 2),
                round(per_request_throughputs[i], 2),
                bottlenecks[i],
                round(compute_times[i], 6),
                round(memory_times[i], 6)
            ))
        
        print("\nThroughput and Bottleneck Data:")
        print(f"{'Batch Size':<12} {'Total Throughput':<20} {'Per-Request':<15} {'Bottleneck':<15} {'Compute Time':<15} {'Memory Time':<15}")
        print("-" * 95)
        for bs, tt, prt, bot, ct, mt in data:
            print(f"{bs:<12} {tt:<20.2f} {prt:<15.2f} {bot:<15} {ct:<15.6f} {mt:<15.6f}")
        
        plt.tight_layout()
        
        # Save the figures to files
        fig.savefig(f"throughput_plots_{self.model.name.replace(' ', '_')}_{self.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png", 
                    dpi=300, bbox_inches='tight')
        fig2.savefig(f"bottleneck_analysis_{self.model.name.replace(' ', '_')}_{self.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png", 
                    dpi=300, bbox_inches='tight')
        
        print(f"\nFigures saved as:")
        print(f"- throughput_plots_{self.model.name.replace(' ', '_')}_{self.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png")
        print(f"- bottleneck_analysis_{self.model.name.replace(' ', '_')}_{self.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png")
        
        return fig, fig2, data


# Create model and hardware objects
llama = ModelParams()  # Default is LLama 3.3 70B
h100 = HardwareParams()  # Default is 4x NVIDIA H100

# Create tokenomics model
model = TokenomicsModel(llama, h100)

# Print basic information
print(f"Model: {llama.name}")
print(f"Total parameters: {llama.total_params / 1e9:.2f} billion")
print(f"Model size: {llama.model_size_bytes / 1e9:.2f} GB")
print()
print(f"Hardware: {h100.name} x {h100.num_gpus}")
print(f"Effective peak performance: {h100.effective_tflops:.2f} TFLOPS")
print(f"Effective memory bandwidth: {h100.effective_memory_bandwidth_GBs:.2f} GB/s")
print(f"Total memory: {h100.total_memory_GB:.2f} GB")

# Generate and show the plots
input_tokens = 2035
output_tokens = 300

throughput_plot, bottleneck_plot, data_points = model.plot_throughput_vs_batch_size(
    input_tokens, output_tokens
)

# Show the plots
plt.tight_layout()
plt.show()