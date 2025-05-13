import math
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class AdvancedTokenomicsModel:
    """
    Advanced model for estimating LLM inference throughput with sophisticated modeling of
    real-world factors like attention algorithms, memory fragmentation, and bandwidth efficiency.
    """
    
    def __init__(self, model_params, hardware_params):
        self.model = model_params
        self.hardware = hardware_params
        
    def calculate_attention_algorithm_flops(self, sequence_length, is_prefill=True):
        """Calculate FLOPs for attention with awareness of different attention algorithms"""
        S = sequence_length
        H = self.model.hidden_size
        A = self.model.num_attention_heads
        
        # Determine which attention algorithm would be used based on sequence length
        # These thresholds would vary by implementation
        use_flash_attention = sequence_length > 512
        use_block_sparse = sequence_length > 8192
        
        if is_prefill:
            if use_flash_attention and not use_block_sparse:
                # Flash Attention is O(N) in memory accesses but still O(N²) in compute
                # It's more efficient but still has quadratic scaling
                # Make this more efficient to match vLLM's implementation
                return 0.3 * (2 * S * S * H + 5 * S * S * A + 2 * S * S * H)  # Changed from 0.4 to 0.3
                
            elif use_block_sparse:
                # Block-sparse attention only computes a subset of the attention matrix
                # vLLM's paged attention appears more efficient than we initially modeled
                sparsity_factor = 0.2  # Changed from 0.3 to 0.2
                return sparsity_factor * (2 * S * S * H + 5 * S * S * A + 2 * S * S * H)
                
            else:
                # Standard attention (full quadratic computation)
                return 2 * S * S * H + 5 * S * S * A + 2 * S * S * H
        else:
            # For decode phase (single new token)
            if use_flash_attention and not use_block_sparse:
                # Flash attention is still linear in decode phase but with better constant factors
                # Make this more efficient
                return 0.5 * (2 * S * H + 5 * S * A + 2 * S * H)  # Changed from 0.7 to 0.5
                
            elif use_block_sparse:
                # Block-sparse in decode can skip some computations
                # vLLM's paged attention is more efficient in decode than we thought
                sparsity_factor = 0.3  # Changed from 0.4 to 0.3
                return sparsity_factor * (2 * S * H + 5 * S * A + 2 * S * H)
                
            else:
                # Standard attention for single token decode
                return 2 * S * H + 5 * S * A + 2 * S * H
    
    def calculate_paged_attention_overhead(self, sequence_length, batch_size=1):
        """Model the overhead from paged attention in vLLM"""
        # Only relevant for long sequences
        if sequence_length < 2048:
            return 0.0
        
        # Page size in vLLM (number of tokens per page)
        page_size = 16  # Adjust based on actual vLLM configuration
        
        # Number of pages needed
        num_pages = math.ceil(sequence_length / page_size)
        
        # Base overhead per page access
        base_overhead = 2e-8
        
        # Overhead increases with the number of pages that need to be accessed
        # This models the page table lookups and indirection
        return base_overhead * num_pages * math.log2(num_pages)
    
    def calculate_prefill_flops(self, sequence_length):
        """Calculate FLOPs for the prefill phase with improved attention modeling"""
        S = sequence_length
        H = self.model.hidden_size
        
        # Use the improved attention algorithm calculation
        attention_flops = self.calculate_attention_algorithm_flops(sequence_length, is_prefill=True)
        
        # The rest stays the same
        # RMS Norm
        attention_flops += 4 * S * H
        
        # Query projection
        attention_flops += 2 * S * H * H
        
        # Key and value projections (LLama specific with reduced KV heads)
        attention_flops += 0.5 * S * H * H
        
        # Positional embedding (RoPE)
        attention_flops += 6 * S * H
        
        # O-Projection
        attention_flops += 2 * S * H * H
        
        # MLP FLOPs
        mlp_flops = 21 * S * H * H
        
        # LM head FLOPs
        lm_head_flops = 2 * S * H * self.model.vocab_size
        
        # Total per layer and final
        per_layer_flops = attention_flops + mlp_flops
        total_flops = (per_layer_flops * self.model.num_hidden_layers) + lm_head_flops
        
        return total_flops
    
    def calculate_decode_flops(self, sequence_length):
        """Calculate FLOPs for a single token in the decode phase with improved attention modeling"""
        H = self.model.hidden_size
        
        # Use the improved attention algorithm calculation
        attention_flops = self.calculate_attention_algorithm_flops(sequence_length, is_prefill=False)
        
        # The rest stays the same
        # RMS Norm (for new token only)
        attention_flops += 4 * H
        
        # Query projection (for new token only)
        attention_flops += 2 * H * H
        
        # Key and value projections (for new token only, LLama specific)
        attention_flops += 0.5 * H * H
        
        # Positional embedding (for new token only)
        attention_flops += 6 * H
        
        # O-Projection (for new token only)
        attention_flops += 2 * H * H
        
        # MLP FLOPs (for new token only)
        mlp_flops = 21 * H * H
        
        # LM head FLOPs (for new token only)
        lm_head_flops = 2 * H * self.model.vocab_size
        
        # Add paged attention overhead - specific to vLLM
        paged_attn_flops = self.calculate_paged_attention_overhead(sequence_length) * 1e12
        
        # Total per layer and final
        per_layer_flops = attention_flops + mlp_flops
        total_flops = (per_layer_flops * self.model.num_hidden_layers) + lm_head_flops + paged_attn_flops
        
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
    
    def calculate_kv_cache_access_penalty(self, sequence_length, batch_size=1):
        """More realistic modeling of KV cache access inefficiency with longer contexts"""
        # Calculate total cache size in GB
        cache_size_GB = self.calculate_kv_cache_size(sequence_length, batch_size) / 1e9
        
        # Use a continuous function instead of discrete thresholds
        # Base penalty starts at 1.0 and increases non-linearly
        # Parameters tuned based on observed vLLM performance
        base_penalty = 1.0
        penalty_factor = 0.015  # Reduced from 0.025 to match observed throughput
        
        # Apply exponential penalty that accelerates as cache size grows
        # This better models cache thrashing and TLB misses
        exponential_factor = 1.2  # Reduced from 1.5 to be less aggressive
        penalty = base_penalty + penalty_factor * (cache_size_GB ** exponential_factor)
        
        # Cap the penalty at a reasonable maximum
        max_penalty = 2.0  # Reduced from 3.5 to be less severe
        return min(penalty, max_penalty)
    
    def memory_fragmentation_factor(self, sequence_length, batch_size=1):
        """Model memory fragmentation effects which increase with sequence length"""
        # Base fragmentation starts at 1.0 (no fragmentation)
        base_factor = 1.0
        
        # Fragmentation starts becoming significant after certain sequence length
        # This threshold depends on the block size used in vLLM
        fragmentation_threshold = 8192  # Increased from 4096 to reflect better paging in vLLM
        
        if sequence_length <= fragmentation_threshold:
            return base_factor
        
        # Model how fragmentation increases with sequence length
        # Coefficient reduced to match observed performance
        fragmentation_factor = base_factor + 0.008 * ((sequence_length - fragmentation_threshold) / 1024)
        
        # Cap at a reasonable maximum - less severe than before
        return min(fragmentation_factor, 1.4)
    
    def effective_memory_bandwidth(self, sequence_length, batch_size=1):
        """Calculate effective memory bandwidth that decreases with longer contexts"""
        # Start with the hardware's effective bandwidth
        base_bandwidth = self.hardware.effective_memory_bandwidth_GBs
        
        # Increase the base efficiency for vLLM's optimized implementation
        # vLLM seems to be more efficient than our original model assumed
        base_adjustment = 1.3  # vLLM is ~30% more efficient than basic models
        adjusted_base = base_bandwidth * base_adjustment
        
        # Calculate how bandwidth efficiency decreases with sequence length
        # This models the transition from sequential to more random access patterns
        # Start reducing efficiency after this length - increased threshold
        efficiency_threshold = 4096  # Increased from 2048
        
        if sequence_length <= efficiency_threshold:
            return adjusted_base
        
        # Model bandwidth efficiency drop - less aggressive reduction
        efficiency_factor = 1.0 - 0.05 * math.log2(sequence_length / efficiency_threshold)  # Reduced from 0.1
        efficiency_factor = max(0.75, efficiency_factor)  # Minimum efficiency increased from 0.6 to 0.75
        
        return adjusted_base * efficiency_factor
    
    def page_table_overhead(self, sequence_length, batch_size=1):
        """Model page table management overhead for long contexts"""
        # Only significant for very long contexts
        if sequence_length < 8192:
            return 0.0
        
        # Calculate overhead based on number of pages needed
        # Assuming 2MB pages and 16 bytes per KV head entry
        bytes_per_position = 2 * self.model.dtype_bytes * self.model.num_hidden_layers * self.model.head_size * self.model.num_key_value_heads
        
        # Page table overhead increases with number of pages
        approx_pages = (sequence_length * bytes_per_position) / (2 * 1024 * 1024)
        base_overhead = 1e-6  # Base overhead per page
        
        return base_overhead * approx_pages
    
    def calculate_comm_overhead(self, data_size_GB, batch_size=1):
        """More realistic communication overhead calculation"""
        if not self.hardware.tensor_parallel or self.hardware.num_gpus <= 1:
            return 0
            
        # Fixed latency cost per communication (3-5 microseconds)
        fixed_latency = 5e-6  # 5 microseconds per communication
        
        # Number of communications (scales with model depth)
        num_comms = 2 * self.model.num_hidden_layers
        
        # Total fixed latency
        total_fixed_latency = fixed_latency * num_comms
        
        # Data transfer time (depends on size and NVLink bandwidth)
        # Add contention factor that increases with batch size
        contention_factor = 1.0 + 0.1 * math.log2(batch_size) if batch_size > 1 else 1.0
        data_transfer_time = (data_size_GB / self.hardware.nvlink_bandwidth_GBs) * contention_factor
        
        return total_fixed_latency + data_transfer_time
    
    def nccl_overhead(self, batch_size):
        """Model NCCL's additional overhead"""
        # Base overhead increases with GPU count
        base_overhead = 1e-5 * (self.hardware.num_gpus - 1)
        
        # Scaling factor based on batch size (larger batches = more data to synchronize)
        scaling_factor = 1.0 + 0.05 * math.log2(batch_size) if batch_size > 1 else 1.0
        
        return base_overhead * scaling_factor
    
    def memory_management_overhead(self, batch_size):
        """Model memory management overhead"""
        # Base overhead for memory operations
        base_overhead = 2e-5
        
        # Increases with batch size due to more complex memory patterns
        return base_overhead * (1.0 + 0.1 * math.log2(batch_size)) if batch_size > 1 else base_overhead
    
    def kernel_launch_overhead(self):
        """Model kernel launch overhead"""
        # Per layer kernel launches
        kernels_per_layer = 10  # Approximate number of kernel launches per layer
        kernel_launch_time = 1e-6  # 1 microsecond per launch
        
        return kernel_launch_time * kernels_per_layer * self.model.num_hidden_layers
    
    def batch_efficiency_factor(self, batch_size):
        """Model how batch efficiency changes with batch size"""
        # vLLM's optimized implementation shows better batch efficiency than we originally modeled
        if batch_size == 1:
            return 1.0
        elif batch_size <= 4:
            return 0.95  # Improved from 0.9 - vLLM handles small batches better
        elif batch_size <= 16:
            return 0.92  # Improved from 0.85
        else:
            # Efficiency decreases with very large batches, but less than we thought
            return 0.9   # Improved from 0.8
    
    def token_processing_variability(self):
        """Model the natural variability in token processing time"""
        # Add a small random factor to account for variations
        return random.uniform(0.95, 1.05)
    
    def prefill_time(self, sequence_length, batch_size=1):
        """Estimate the prefill time for processing the input prompt"""
        total_flops = self.calculate_prefill_flops(sequence_length) * batch_size
        total_tflops = total_flops / 1e12
        
        # Apply batch efficiency factor
        batch_factor = self.batch_efficiency_factor(batch_size)
        compute_time = (total_tflops / self.hardware.effective_tflops) / batch_factor
        
        # Communication overhead with enhanced model
        comm_size_bytes = batch_size * sequence_length * self.model.hidden_size * self.model.dtype_bytes
        comm_size_GB = comm_size_bytes / 1e9
        comm_overhead = self.calculate_comm_overhead(comm_size_GB, batch_size)
        
        # Add NCCL overhead
        nccl_overhead = self.nccl_overhead(batch_size)
        
        # Add kernel launch overhead
        kernel_overhead = self.kernel_launch_overhead()
        
        # Add memory management overhead
        mem_mgmt_overhead = self.memory_management_overhead(batch_size)
        
        # Add natural variability
        variability = self.token_processing_variability()
        
        total_time = (compute_time + comm_overhead + nccl_overhead + kernel_overhead + mem_mgmt_overhead) * variability
        
        return total_time
    
    def decode_time_breakdown(self, sequence_length, batch_size=1):
        """Estimate the compute and memory time for a single decode step with improved long-context modeling"""
        # Calculate memory access time (loading model weights + KV cache)
        model_size_GB = self.model.model_size_bytes / 1e9
        kv_cache_size_GB = self.calculate_kv_cache_size(sequence_length, batch_size) / 1e9
        
        # Apply more sophisticated KV cache access penalty
        kv_cache_penalty = self.calculate_kv_cache_access_penalty(sequence_length, batch_size)
        
        # Apply memory fragmentation factor
        fragmentation_factor = self.memory_fragmentation_factor(sequence_length, batch_size)
        
        # Get adjusted memory bandwidth for long contexts
        effective_bandwidth = self.effective_memory_bandwidth(sequence_length, batch_size)
        
        # Calculate memory access time with more realistic modeling
        memory_load_time = (model_size_GB + kv_cache_size_GB * kv_cache_penalty * fragmentation_factor) / effective_bandwidth
        
        # Calculate compute time with batch efficiency factor
        decode_flops = self.calculate_decode_flops(sequence_length) * batch_size
        decode_tflops = decode_flops / 1e12
        batch_factor = self.batch_efficiency_factor(batch_size)
        compute_time = (decode_tflops / self.hardware.effective_tflops) / batch_factor
        
        # Calculate all overhead components
        comm_size_bytes = batch_size * self.model.hidden_size * self.model.dtype_bytes
        comm_size_GB = comm_size_bytes / 1e9
        comm_overhead = self.calculate_comm_overhead(comm_size_GB, batch_size)
        nccl_overhead = self.nccl_overhead(batch_size)
        mem_mgmt_overhead = self.memory_management_overhead(batch_size)
        kernel_overhead = self.kernel_launch_overhead()
        
        # Add page table overhead for very long contexts
        page_overhead = self.page_table_overhead(sequence_length, batch_size)
        
        # Add attention algorithm overhead - this models the crossover point where naive attention
        # becomes more expensive than flash attention or other specialized algorithms
        attn_algo_overhead = 0.0
        if sequence_length > 8192:
            # Model flash attention crossover - when sequence length becomes very large,
            # even optimized algorithms slow down somewhat
            attn_algo_overhead = 1e-6 * (sequence_length - 8192)
        
        # Combine all overhead
        total_overhead = (
            comm_overhead + 
            nccl_overhead + 
            mem_mgmt_overhead + 
            kernel_overhead + 
            page_overhead + 
            attn_algo_overhead
        )
        
        # Add natural variability
        variability = self.token_processing_variability()
        
        # Total time is dictated by the slower of compute or memory, plus total overhead
        total_time = (max(compute_time, memory_load_time) + total_overhead) * variability
        
        # Determine the bottleneck
        bottleneck = "memory" if memory_load_time > compute_time else "compute"
        
        # For very long contexts, add a "long context efficiency" factor
        # This models vLLM's optimizations for long context handling
        if sequence_length > 8192:
            # vLLM seems to perform better than naive models would predict for long contexts
            long_context_factor = 1.0 - 0.02 * math.log2(sequence_length / 8192)
            long_context_factor = max(0.85, long_context_factor)  # Don't let it go below 0.85
            total_time *= long_context_factor
        
        return {
            "compute_time": compute_time,
            "memory_time": memory_load_time,
            "memory_bandwidth_GBs": effective_bandwidth,
            "kv_cache_penalty": kv_cache_penalty,
            "fragmentation_factor": fragmentation_factor,
            "comm_overhead": comm_overhead,
            "nccl_overhead": nccl_overhead,
            "mem_mgmt_overhead": mem_mgmt_overhead,
            "kernel_overhead": kernel_overhead,
            "page_overhead": page_overhead,
            "attn_algo_overhead": attn_algo_overhead,
            "total_overhead": total_overhead,
            "bottleneck": bottleneck,
            "total_time": total_time
        }
    
    def decode_time(self, sequence_length, batch_size=1, generated_tokens=1):
        """Estimate the decode time for generating tokens after the prefill"""
        # Initialize total time
        total_decode_time = 0
        
        # Current sequence length starts at prompt length
        current_sequence_length = sequence_length
        
        # For each token to generate - ensure generated_tokens is an integer
        for _ in range(int(generated_tokens)):
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
    
    def plot_context_length_scaling(self, output_tokens=100, batch_size=1, context_lengths=None):
        """
        Plot how throughput and decoding time scale with increasing context length
        
        Args:
            output_tokens: Number of tokens to generate
            batch_size: Batch size to use
            context_lengths: List of context lengths to evaluate
        """
        if context_lengths is None:
            # Default range of context lengths to analyze
            context_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        
        # Calculate metrics for each context length
        throughputs = []
        decode_times = []
        kv_penalties = []
        fragmentation_factors = []
        bandwidth_efficiencies = []
        bottlenecks = []
        
        for context_length in context_lengths:
            # Calculate throughput
            throughput = self.calculate_per_request_throughput(context_length, output_tokens, batch_size)
            throughputs.append(throughput)
            
            # Calculate decode time for the first generated token
            breakdown = self.decode_time_breakdown(context_length, batch_size)
            decode_times.append(breakdown["total_time"])
            bottlenecks.append(breakdown["bottleneck"])
            
            # Gather factors
            kv_penalties.append(breakdown.get("kv_cache_penalty", 1.0))
            fragmentation_factors.append(breakdown.get("fragmentation_factor", 1.0))
            
            # Calculate bandwidth efficiency 
            base_bandwidth = self.hardware.memory_bandwidth_GBs * self.hardware.memory_utilization
            actual_bandwidth = breakdown.get("memory_bandwidth_GBs", base_bandwidth)
            bandwidth_efficiencies.append(actual_bandwidth / base_bandwidth)
        
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot throughput vs context length
        axs[0, 0].plot(context_lengths, throughputs, 'bo-', linewidth=2, markersize=8)
        axs[0, 0].set_xlabel('Context Length')
        axs[0, 0].set_ylabel('Throughput (tokens/second)')
        axs[0, 0].set_title('Throughput vs Context Length')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].set_xscale('log', base=2)
        
        # Add throughput values above points
        for i, v in enumerate(throughputs):
            axs[0, 0].text(context_lengths[i], v * 1.05, f"{v:.1f}", ha='center', fontsize=9)
        
        # Plot decode time vs context length
        axs[0, 1].plot(context_lengths, decode_times, 'ro-', linewidth=2, markersize=8)
        axs[0, 1].set_xlabel('Context Length')
        axs[0, 1].set_ylabel('Decode Time (seconds)')
        axs[0, 1].set_title('Decode Time per Token vs Context Length')
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].set_xscale('log', base=2)
        
        # Add time values above points
        for i, v in enumerate(decode_times):
            axs[0, 1].text(context_lengths[i], v * 1.05, f"{v*1000:.1f}ms", ha='center', fontsize=9)
        
        # Plot KV cache penalty and fragmentation factor
        axs[1, 0].plot(context_lengths, kv_penalties, 'go-', linewidth=2, markersize=8, label='KV Cache Penalty')
        axs[1, 0].plot(context_lengths, fragmentation_factors, 'mo-', linewidth=2, markersize=8, label='Fragmentation Factor')
        axs[1, 0].set_xlabel('Context Length')
        axs[1, 0].set_ylabel('Factor Value')
        axs[1, 0].set_title('Memory Penalty Factors vs Context Length')
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].set_xscale('log', base=2)
        axs[1, 0].legend()
        
        # Plot bandwidth efficiency
        axs[1, 1].plot(context_lengths, bandwidth_efficiencies, 'co-', linewidth=2, markersize=8)
        axs[1, 1].set_xlabel('Context Length')
        axs[1, 1].set_ylabel('Bandwidth Efficiency')
        axs[1, 1].set_title('Memory Bandwidth Efficiency vs Context Length')
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].set_xscale('log', base=2)
        
        # Add efficiency values above points
        for i, v in enumerate(bandwidth_efficiencies):
            axs[1, 1].text(context_lengths[i], v * 1.05, f"{v*100:.1f}%", ha='center', fontsize=9)
        
        # Mark bottlenecks on the throughput plot
        for i, bottleneck in enumerate(bottlenecks):
            marker_color = 'blue' if bottleneck == 'compute' else 'red'
            axs[0, 0].plot(context_lengths[i], throughputs[i], 'o', markersize=12, 
                        markerfacecolor='none', markeredgecolor=marker_color, markeredgewidth=2)
        
        # Add a legend for bottlenecks
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='blue', 
                  markeredgewidth=2, markersize=10, label='Compute Bound'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='red', 
                  markeredgewidth=2, markersize=10, label='Memory Bound')
        ]
        axs[0, 0].legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f'Context Length Scaling Analysis: {self.model.name} on {self.hardware.num_gpus}x {self.hardware.name}', 
                    fontsize=16)
        
        return fig
    
    def plot_throughput_comparison(self, input_tokens, output_tokens, actual_results, batch_sizes=None):
        """
        Plot theoretical vs actual throughput against batch size with bottleneck analysis
        
        Args:
            actual_results: Dictionary with batch sizes as keys and results as values
        """
        if batch_sizes is None:
            # Use the batch sizes from the actual results
            batch_sizes = sorted([int(k) for k in actual_results.keys()])
        
        # Calculate theoretical throughputs
        theoretical_total_throughputs = []
        theoretical_per_request_throughputs = []
        
        # Analyze bottlenecks
        bottleneck_analysis = self.analyze_bottlenecks(input_tokens, output_tokens, batch_sizes)
        compute_times = bottleneck_analysis["compute_times"]
        memory_times = bottleneck_analysis["memory_times"]
        bottlenecks = bottleneck_analysis["bottlenecks"]
        
        # Get actual throughputs
        actual_total_throughputs = []
        actual_per_request_throughputs = []
        
        for batch_size in batch_sizes:
            # Theoretical throughputs
            total_throughput = self.calculate_total_throughput(input_tokens, output_tokens, batch_size)
            per_request_throughput = self.calculate_per_request_throughput(input_tokens, output_tokens, batch_size)
            
            theoretical_total_throughputs.append(total_throughput)
            theoretical_per_request_throughputs.append(per_request_throughput)
            
            # Actual throughputs (convert batch_size to string for dictionary lookup)
            bs_key = str(batch_size)
            if bs_key in actual_results:
                actual_total_throughputs.append(actual_results[bs_key]["tokens_per_second_in_batch"])
                actual_per_request_throughputs.append(actual_results[bs_key]["avg_tokens_per_second"])
            else:
                actual_total_throughputs.append(None)
                actual_per_request_throughputs.append(None)
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Format GPU info for title
        gpu_info = f"{self.hardware.num_gpus}x{self.hardware.name}"
        
        # Plot total throughput (theoretical and actual)
        ax1.plot(batch_sizes, theoretical_total_throughputs, 'bo-', linewidth=2, markersize=8, label='Theoretical')
        ax1.plot(batch_sizes, actual_total_throughputs, 'go-', linewidth=2, markersize=8, label='Actual')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Total Throughput (tokens/second)')
        ax1.set_title(f'Total Throughput vs Batch Size\n{self.model.name} on {gpu_info}')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)  # Use log scale for x-axis
        ax1.set_xticks(batch_sizes)
        ax1.set_xticklabels(batch_sizes)
        ax1.legend()
        
        # Add actual token counts above points for total throughput
        for i, v in enumerate(actual_total_throughputs):
            if v is not None:
                ax1.text(batch_sizes[i], v + (max(actual_total_throughputs) * 0.03), 
                        f"{v:.1f} t/s", ha='center', va='bottom', fontsize=9)
        
        # Plot per-request throughput (theoretical and actual)
        ax2.plot(batch_sizes, theoretical_per_request_throughputs, 'ro-', linewidth=2, markersize=8, label='Theoretical')
        ax2.plot(batch_sizes, actual_per_request_throughputs, 'mo-', linewidth=2, markersize=8, label='Actual')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Per-Request Throughput (tokens/second)')
        ax2.set_title(f'Per-Request Throughput vs Batch Size\n{self.model.name} on {gpu_info}')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)  # Use log scale for x-axis
        ax2.set_xticks(batch_sizes)
        ax2.set_xticklabels(batch_sizes)
        ax2.legend()
        
        # Add actual token counts above points for per-request throughput
        for i, v in enumerate(actual_per_request_throughputs):
            if v is not None:
                ax2.text(batch_sizes[i], v + (max(actual_per_request_throughputs) * 0.03), 
                        f"{v:.1f} t/s", ha='center', va='bottom', fontsize=9)
        
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
        ax3.set_title(f'Compute vs Memory Bottleneck Analysis\n{self.model.name} on {gpu_info}')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)  # Use log scale for x-axis
        ax3.set_xticks(batch_sizes)
        ax3.set_xticklabels(batch_sizes)
        ax3.legend()
        
        # Find crossover point
        for i in range(len(batch_sizes) - 1):
            if i < len(bottlenecks) - 1 and bottlenecks[i] != bottlenecks[i+1]:
                crossover_point = batch_sizes[i+1]
                ax3.axvline(x=crossover_point, color='green', linestyle='--', 
                        label=f'Bottleneck Shift at Batch Size {crossover_point}')
                ax3.legend()
                break
        
        # Create a third figure for the relative efficiency
        fig3, ax4 = plt.subplots(figsize=(12, 7))
        
        # Calculate efficiency (actual / theoretical)
        efficiencies = []
        for i, batch_size in enumerate(batch_sizes):
            if actual_total_throughputs[i] is None or theoretical_total_throughputs[i] == 0:
                efficiencies.append(None)
            else:
                efficiency = actual_total_throughputs[i] / theoretical_total_throughputs[i] * 100
                efficiencies.append(efficiency)
        
        # Plot efficiency
        ax4.plot(batch_sizes, efficiencies, 'ko-', linewidth=2, markersize=8)
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Efficiency (Actual / Theoretical) %')
        ax4.set_title(f'Inference Efficiency vs Batch Size\n{self.model.name} on {gpu_info}')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)  # Use log scale for x-axis
        ax4.set_xticks(batch_sizes)
        ax4.set_xticklabels(batch_sizes)
        
        # Add values above points in efficiency plot
        for i, v in enumerate(efficiencies):
            if v is not None:
                ax4.text(batch_sizes[i], v + 2, f"{v:.1f}%", 
                        ha='center', va='bottom', fontweight='bold')
        
        # Create a data table with the values and bottlenecks
        data = []
        for i, bs in enumerate(batch_sizes):
            bs_key = str(bs)
            if bs_key in actual_results:
                actual_total = actual_results[bs_key]["tokens_per_second_in_batch"]
                actual_per_request = actual_results[bs_key]["avg_tokens_per_second"]
                efficiency = (actual_total / theoretical_total_throughputs[i]) * 100 if theoretical_total_throughputs[i] > 0 else 0
            else:
                actual_total = None
                actual_per_request = None
                efficiency = None
                
            data.append((
                bs, 
                round(theoretical_total_throughputs[i], 2),
                actual_total if actual_total is None else round(actual_total, 2),
                round(efficiency, 2) if efficiency is not None else None,
                bottlenecks[i],
                round(compute_times[i], 6),
                round(memory_times[i], 6)
            ))
        
        print("\nThroughput and Bottleneck Data:")
        print(f"{'Batch Size':<12} {'Theo. Total':<15} {'Actual Total':<15} {'Efficiency %':<15} {'Bottleneck':<15} {'Compute Time':<15} {'Memory Time':<15}")
        print("-" * 110)
        for bs, tt, at, eff, bot, ct, mt in data:
            at_str = f"{at:.2f}" if at is not None else "N/A"
            eff_str = f"{eff:.2f}" if eff is not None else "N/A"
            print(f"{bs:<12} {tt:<15.2f} {at_str:<15} {eff_str:<15} {bot:<15} {ct:<15.6f} {mt:<15.6f}")
        
        plt.tight_layout()
        return fig, fig2, fig3, data