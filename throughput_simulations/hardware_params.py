from dataclasses import dataclass

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
    compute_utilization: float = 0.6  # Typical compute utilization rate (60%)
    memory_utilization: float = 0.7  # Typical memory bandwidth utilization (70%)
    
    def __post_init__(self):
        # Total memory across all GPUs
        self.total_memory_GB = self.memory_size_GB * self.num_gpus
        
        # Effective FLOPS and bandwidth when using multiple GPUs
        if self.tensor_parallel:
            # In tensor parallelism, we use all GPUs simultaneously
            self.effective_tflops = self.tflops * self.num_gpus * self.compute_utilization
            self.effective_memory_bandwidth_GBs = self.memory_bandwidth_GBs * self.num_gpus * self.memory_utilization
        else:
            # In pipeline parallelism, we use GPUs one after another (for throughput, not latency)
            self.effective_tflops = self.tflops * self.compute_utilization
            self.effective_memory_bandwidth_GBs = self.memory_bandwidth_GBs * self.memory_utilization