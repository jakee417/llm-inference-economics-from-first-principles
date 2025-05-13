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