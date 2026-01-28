"""
Interactive Dashboard for LLM Inference Economics Simulator
"""

import streamlit as st
import numpy as np
import pandas as pd
from model_params import ModelParams
from hardware_params import HardwareParams
from tokenomics_model import TokenomicsModel
from advanced_tokenomics_model import AdvancedTokenomicsModel

st.set_page_config(
    page_title="LLM Inference Economics Dashboard",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state for sweep analysis
if 'sweep_variable' not in st.session_state:
    st.session_state.sweep_variable = None

def set_sweep_variable(var_name):
    st.session_state.sweep_variable = var_name

def clear_sweep():
    st.session_state.sweep_variable = None

def reset_to_defaults():
    """Reset all parameters to their default values."""
    # Clear sweep mode
    st.session_state.sweep_variable = None
    # Set all widget states to default values
    st.session_state['num_gpus_slider'] = 4
    st.session_state['compute_util_slider'] = 0.6
    st.session_state['mem_util_slider'] = 0.7
    st.session_state['input_tokens_slider'] = 2048
    st.session_state['output_tokens_slider'] = 300
    st.session_state['batch_size_slider'] = 1
    st.session_state['gpu_model_select'] = "NVIDIA H100 SXM"
    st.session_state['model_select'] = "Llama 3.3 70B"
    st.session_state['tensor_parallel_checkbox'] = True
    st.session_state['advanced_model_checkbox'] = True

# Default values for reference
DEFAULTS = {
    "gpu_model": "NVIDIA H100 SXM",
    "num_gpus": 4,
    "tensor_parallel": True,
    "compute_utilization": 0.6,
    "memory_utilization": 0.7,
    "model": "Llama 3.3 70B",
    "input_tokens": 2048,
    "output_tokens": 300,
    "batch_size": 1,
    "use_advanced_model": True
}

# === Define sweep configurations for each variable ===
# Values match the slider ranges
SWEEP_CONFIGS = {
    "num_gpus": {
        "name": "Number of GPUs",
        "values": [1, 2, 3, 4, 5, 6, 7, 8],  # Full slider range (1-8, step 1)
        "param_key": "num_gpus",
        "category": "hardware"
    },
    "compute_utilization": {
        "name": "Compute Utilization",
        "values": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],  # Full slider range (0.1-1.0, step 0.05)
        "param_key": "compute_utilization",
        "category": "hardware"
    },
    "memory_utilization": {
        "name": "Memory Utilization",
        "values": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],  # Full slider range (0.1-1.0, step 0.05)
        "param_key": "memory_utilization",
        "category": "hardware"
    },
    "input_tokens": {
        "name": "Input Tokens",
        "values": [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],  # Logarithmic sampling across slider range (32-32768)
        "param_key": "input_tokens",
        "category": "inference"
    },
    "output_tokens": {
        "name": "Output Tokens",
        "values": [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],  # Logarithmic sampling across slider range (1-4096)
        "param_key": "output_tokens",
        "category": "inference"
    },
    "batch_size": {
        "name": "Batch Size",
        "values": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],  # Exact select_slider options
        "param_key": "batch_size",
        "category": "inference"
    }
}

def compute_metrics(tokenomics, input_tokens, output_tokens, batch_size, hardware):
    """Compute all metrics for a given configuration."""
    prefill_time = tokenomics.prefill_time(input_tokens, batch_size)
    decode_time_total = tokenomics.decode_time(input_tokens, batch_size, output_tokens)
    total_time = tokenomics.total_inference_time(input_tokens, output_tokens, batch_size)
    total_throughput = tokenomics.calculate_total_throughput(input_tokens, output_tokens, batch_size)
    per_request_throughput = tokenomics.calculate_per_request_throughput(input_tokens, output_tokens, batch_size)
    breakdown = tokenomics.decode_time_breakdown(input_tokens + output_tokens, batch_size)
    kv_cache_bytes = tokenomics.calculate_kv_cache_size(input_tokens + output_tokens, batch_size)

    compute_memory_ratio = breakdown['compute_time'] / breakdown['memory_time'] if breakdown['memory_time'] > 0 else float('inf')

    # Calculate model vs KV cache load time breakdown (decode phase)
    model_size_gb = tokenomics.model.model_size_bytes / 1e9
    kv_cache_size_gb = tokenomics.calculate_kv_cache_size(input_tokens + output_tokens, batch_size) / 1e9

    # Get effective bandwidth (use the one from breakdown if available, otherwise calculate)
    if hasattr(tokenomics, 'effective_memory_bandwidth'):
        effective_bw = tokenomics.effective_memory_bandwidth(input_tokens, batch_size)
    else:
        effective_bw = hardware.effective_memory_bandwidth_GBs

    model_load_time_ms = (model_size_gb / effective_bw) * 1000
    kv_cache_load_time_ms = (kv_cache_size_gb / effective_bw) * 1000

    return {
        "total_throughput": total_throughput,
        "per_request_throughput": per_request_throughput,
        "total_time": total_time,
        "prefill_time_ms": prefill_time * 1000,
        "decode_time": decode_time_total,
        "avg_time_per_token_ms": (decode_time_total / output_tokens) * 1000 if output_tokens > 0 else 0,
        "compute_time_ms": breakdown['compute_time'] * 1000,
        "memory_time_ms": breakdown['memory_time'] * 1000,
        "model_load_time_ms": model_load_time_ms,
        "kv_cache_load_time_ms": kv_cache_load_time_ms,
        "compute_memory_ratio": compute_memory_ratio,
        "kv_cache_gb": kv_cache_bytes / 1e9,
        "model_size_gb": model_size_gb,
        "bottleneck": breakdown['bottleneck'],
        "comm_overhead_ms": breakdown.get('comm_overhead', 0) * 1000
    }

def run_sweep_analysis(sweep_var, base_config, model_choice, gpu_choice, gpu_presets, use_advanced_model):
    """Run a sweep analysis for a given variable."""
    config = SWEEP_CONFIGS[sweep_var]
    sweep_values = config["values"]
    results = []

    for val in sweep_values:
        try:
            # Create modified config
            hw_params = base_config["hardware"].copy()
            inf_params = base_config["inference"].copy()

            if config["category"] == "hardware":
                hw_params[config["param_key"]] = val
            else:
                inf_params[config["param_key"]] = val

            # Create hardware object
            if gpu_choice != "Custom":
                preset = gpu_presets[gpu_choice]
                hw = HardwareParams(
                    name=gpu_choice,
                    tflops=preset["tflops"],
                    memory_bandwidth_GBs=preset["memory_bandwidth_GBs"],
                    memory_size_GB=preset["memory_size_GB"],
                    num_gpus=hw_params["num_gpus"],
                    tensor_parallel=hw_params["tensor_parallel"],
                    nvlink_bandwidth_GBs=preset["nvlink_bandwidth_GBs"],
                    compute_utilization=hw_params["compute_utilization"],
                    memory_utilization=hw_params["memory_utilization"]
                )
            else:
                hw = HardwareParams(
                    name="Custom GPU",
                    tflops=hw_params["tflops"],
                    memory_bandwidth_GBs=hw_params["memory_bandwidth_GBs"],
                    memory_size_GB=hw_params["memory_size_GB"],
                    num_gpus=hw_params["num_gpus"],
                    tensor_parallel=hw_params["tensor_parallel"],
                    nvlink_bandwidth_GBs=hw_params["nvlink_bandwidth_GBs"],
                    compute_utilization=hw_params["compute_utilization"],
                    memory_utilization=hw_params["memory_utilization"]
                )

            # Create model object
            if model_choice == "Custom":
                mdl = ModelParams(
                    name="custom",
                    hidden_size=base_config["model"]["hidden_size"],
                    num_hidden_layers=base_config["model"]["num_hidden_layers"],
                    num_attention_heads=base_config["model"]["num_attention_heads"],
                    num_key_value_heads=base_config["model"]["num_key_value_heads"],
                    intermediate_size=base_config["model"]["intermediate_size"],
                    vocab_size=base_config["model"]["vocab_size"],
                    dtype_bytes=base_config["model"]["dtype_bytes"]
                )
            else:
                mdl = ModelParams(name=base_config["model"]["name"], dtype_bytes=base_config["model"]["dtype_bytes"])

            # Create tokenomics model
            if use_advanced_model:
                tok = AdvancedTokenomicsModel(mdl, hw)
            else:
                tok = TokenomicsModel(mdl, hw)

            # Compute metrics
            metrics = compute_metrics(
                tok,
                inf_params["input_tokens"],
                inf_params["output_tokens"],
                inf_params["batch_size"],
                hw
            )
            metrics["sweep_value"] = val
            results.append(metrics)

        except Exception as e:
            # Skip values that cause errors (e.g., OOM)
            continue

    return results

# Main app
st.title("LLM Inference Economics Dashboard")
st.markdown("*Analyze throughput, efficiency, and bottlenecks for LLM inference*")

# Sidebar for all input parameters
st.sidebar.header("Configuration")

# Reset button at the top
if st.sidebar.button("Reset to Defaults", use_container_width=True, help="Reset all parameters to their default values"):
    reset_to_defaults()
    st.rerun()

st.sidebar.divider()

# === Hardware Parameters ===
st.sidebar.subheader("Hardware Parameters")

gpu_presets = {
    "NVIDIA H100 SXM": {"tflops": 989, "memory_bandwidth_GBs": 3350, "memory_size_GB": 80, "nvlink_bandwidth_GBs": 450},
    "NVIDIA H100 PCIe": {"tflops": 756, "memory_bandwidth_GBs": 2000, "memory_size_GB": 80, "nvlink_bandwidth_GBs": 450},
    "NVIDIA A100 80GB": {"tflops": 312, "memory_bandwidth_GBs": 2039, "memory_size_GB": 80, "nvlink_bandwidth_GBs": 300},
    "NVIDIA A100 40GB": {"tflops": 312, "memory_bandwidth_GBs": 1555, "memory_size_GB": 40, "nvlink_bandwidth_GBs": 300},
    "NVIDIA L40S": {"tflops": 366, "memory_bandwidth_GBs": 864, "memory_size_GB": 48, "nvlink_bandwidth_GBs": 0},
    "NVIDIA RTX 4090": {"tflops": 330, "memory_bandwidth_GBs": 1008, "memory_size_GB": 24, "nvlink_bandwidth_GBs": 0},
    "NVIDIA RTX 5060 Laptop": {"tflops": 194, "memory_bandwidth_GBs": 256, "memory_size_GB": 8, "nvlink_bandwidth_GBs": 0},
    "Custom": None
}

gpu_choice = st.sidebar.selectbox(
    "GPU Model",
    list(gpu_presets.keys()),
    index=0,
    key="gpu_model_select",
    help="Select a GPU model with preset specs (TFLOPs, memory bandwidth, VRAM). Choose 'Custom' to manually specify hardware parameters. Specs are for FP16/BF16 tensor core operations."
)

if gpu_choice != "Custom":
    preset = gpu_presets[gpu_choice]
    tflops = preset["tflops"]
    memory_bandwidth_GBs = preset["memory_bandwidth_GBs"]
    memory_size_GB = preset["memory_size_GB"]
    nvlink_bandwidth_GBs = preset["nvlink_bandwidth_GBs"]
else:
    tflops = st.sidebar.number_input(
        "Peak TFLOPs", min_value=1.0, max_value=2000.0, value=989.0,
        help="Theoretical peak compute performance in TeraFLOPs for FP16/BF16 tensor operations. Found in GPU spec sheets."
    )
    memory_bandwidth_GBs = st.sidebar.number_input(
        "Memory Bandwidth (GB/s)", min_value=100.0, max_value=10000.0, value=3350.0,
        help="Peak HBM memory bandwidth in GB/s. Determines how fast model weights and KV cache can be loaded."
    )
    memory_size_GB = st.sidebar.number_input(
        "Memory Size (GB)", min_value=8.0, max_value=320.0, value=80.0,
        help="VRAM per GPU in GB. Must fit model weights, KV cache, and activations."
    )
    nvlink_bandwidth_GBs = st.sidebar.number_input(
        "NVLink Bandwidth (GB/s)", min_value=0.0, max_value=1000.0, value=450.0,
        help="Inter-GPU communication bandwidth for tensor parallelism. Set to 0 for GPUs without NVLink (uses PCIe instead)."
    )

# Hardware sliders with sweep buttons below
num_gpus = st.sidebar.slider(
    "Number of GPUs",
    min_value=1, max_value=8, value=4,
    key="num_gpus_slider",
    help="Number of GPUs used for inference. With tensor parallelism, model layers are split across GPUs, increasing both compute capacity and memory bandwidth proportionally."
)
st.sidebar.button("Sweep Number of GPUs", key="sweep_num_gpus", on_click=set_sweep_variable, args=("num_gpus",), use_container_width=True)

tensor_parallel = st.sidebar.checkbox(
    "Tensor Parallelism",
    value=True,
    key="tensor_parallel_checkbox",
    help="ENABLED: Model layers are split across GPUs and computed in parallel. Multiplies effective TFLOPs and memory bandwidth by GPU count, but adds inter-GPU communication overhead. Best for reducing latency. | DISABLED: Assumes pipeline parallelism where different layers reside on different GPUs and execute sequentially. Uses single-GPU compute/bandwidth (no scaling with GPU count). Better for throughput with multiple concurrent requests, but doesn't reduce single-request latency."
)

compute_utilization = st.sidebar.slider(
    "Compute Utilization",
    min_value=0.1, max_value=1.0, value=0.6, step=0.05,
    key="compute_util_slider",
    help="Fraction of peak GPU compute (TFLOPs) actually achieved during inference. Real-world utilization is typically 50-70% due to memory stalls, kernel launch overhead, and pipeline bubbles."
)
st.sidebar.button("Sweep Compute Utilization", key="sweep_compute_util", on_click=set_sweep_variable, args=("compute_utilization",), use_container_width=True)

memory_utilization = st.sidebar.slider(
    "Memory Utilization",
    min_value=0.1, max_value=1.0, value=0.7, step=0.05,
    key="mem_util_slider",
    help="Fraction of peak memory bandwidth actually achieved. Limited by access patterns, cache efficiency, and memory controller saturation. Typically 60-80% for well-optimized inference."
)
st.sidebar.button("Sweep Memory Utilization", key="sweep_mem_util", on_click=set_sweep_variable, args=("memory_utilization",), use_container_width=True)

# === Model Parameters ===
st.sidebar.subheader("Model Parameters")

model_presets = {
    "Llama 3.3 70B": "llama-3-3-70b",
    "Llama 3.1 8B": "llama-3-1-8b",
    "Custom": None
}

model_choice = st.sidebar.selectbox(
    "Model",
    list(model_presets.keys()),
    index=0,
    key="model_select",
    help="Select a preset model architecture or choose Custom to define your own. Preset models use Llama-style architectures with GQA (Grouped Query Attention) and SwiGLU activation."
)

if model_choice == "Custom":
    hidden_size = st.sidebar.number_input(
        "Hidden Size", min_value=256, max_value=16384, value=4096, step=256,
        help="Dimension of the hidden states (d_model). Larger values increase model capacity but also memory and compute requirements quadratically for attention."
    )
    num_hidden_layers = st.sidebar.number_input(
        "Number of Layers", min_value=1, max_value=200, value=32,
        help="Number of transformer layers. Each layer contains attention and MLP blocks. More layers increase model depth and capacity."
    )
    num_attention_heads = st.sidebar.number_input(
        "Attention Heads", min_value=1, max_value=128, value=32,
        help="Number of attention heads for multi-head attention. Head dimension = hidden_size / num_heads. More heads can capture diverse attention patterns."
    )
    num_key_value_heads = st.sidebar.number_input(
        "KV Heads (GQA)", min_value=1, max_value=128, value=8,
        help="Number of key-value heads for Grouped Query Attention (GQA). Fewer KV heads than query heads reduces KV cache size and memory bandwidth requirements."
    )
    intermediate_size = st.sidebar.number_input(
        "Intermediate Size", min_value=256, max_value=65536, value=14336, step=256,
        help="Hidden dimension of the MLP/FFN layers. Typically 2.5-4x the hidden size. Larger values increase model capacity."
    )
    vocab_size = st.sidebar.number_input(
        "Vocabulary Size", min_value=1000, max_value=500000, value=128256,
        help="Number of tokens in the vocabulary. Affects embedding and LM head parameter count and compute."
    )
    dtype_bytes = st.sidebar.selectbox(
        "Data Type",
        [("bfloat16", 2), ("float16", 2), ("float32", 4), ("fp8", 1), ("int8", 1), ("int4", 0.5)],
        format_func=lambda x: x[0],
        help="Numerical precision for model weights. bfloat16/float16 use 2 bytes, float32 uses 4 bytes, fp8/int8 use 1 byte, int4 uses 0.5 bytes per parameter. Lower precision reduces memory and increases throughput but may impact accuracy."
    )[1]
    model_name = "custom"
else:
    model_name = model_presets[model_choice]

# Data type selector (applies to all models)
dtype_options = [("bfloat16", 2), ("float16", 2), ("float32", 4), ("fp8", 1), ("int8", 1), ("int4", 0.5)]
if model_choice != "Custom":
    dtype_bytes = st.sidebar.selectbox(
        "Data Type",
        dtype_options,
        format_func=lambda x: x[0],
        help="Numerical precision for model weights. bfloat16/float16 use 2 bytes, float32 uses 4 bytes, fp8/int8 use 1 byte, int4 uses 0.5 bytes per parameter. Lower precision reduces memory and increases throughput but may impact accuracy."
    )[1]

# === Inference Settings ===
st.sidebar.subheader("Inference Settings")

input_tokens = st.sidebar.slider(
    "Input Tokens (Prompt Length)",
    min_value=32, max_value=32768, value=2048, step=32,
    key="input_tokens_slider",
    help="Number of tokens in the input prompt. Processed during the prefill phase with O(S^2) attention complexity. Longer prompts increase prefill time quadratically and KV cache size linearly."
)
st.sidebar.button("Sweep Input Tokens", key="sweep_input_tokens", on_click=set_sweep_variable, args=("input_tokens",), use_container_width=True)

output_tokens = st.sidebar.slider(
    "Output Tokens",
    min_value=1, max_value=4096, value=300, step=1,
    key="output_tokens_slider",
    help="Number of tokens to generate autoregressively. Each token requires a decode step that loads model weights and attends to all previous tokens. Total decode time scales roughly linearly with output length."
)
st.sidebar.button("Sweep Output Tokens", key="sweep_output_tokens", on_click=set_sweep_variable, args=("output_tokens",), use_container_width=True)

batch_size = st.sidebar.select_slider(
    "Batch Size",
    options=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    value=1,
    key="batch_size_slider",
    help="Number of sequences processed in parallel. Larger batches amortize model weight loading across more tokens, improving throughput but increasing latency and KV cache memory. Critical for transitioning from memory-bound to compute-bound operation."
)
st.sidebar.button("Sweep Batch Size", key="sweep_batch_size", on_click=set_sweep_variable, args=("batch_size",), use_container_width=True)

# === Model Selection ===
st.sidebar.subheader("Simulation Model")
use_advanced_model = st.sidebar.checkbox(
    "Use Advanced Model",
    value=True,
    key="advanced_model_checkbox",
    help="Advanced model includes realistic factors: KV cache access penalties, memory fragmentation, Flash Attention optimization, batch efficiency degradation, NCCL/kernel overheads, and long-context efficiency adjustments. Disable for idealized theoretical estimates."
)

# === Create Parameter Objects ===
hardware = HardwareParams(
    name=gpu_choice if gpu_choice != "Custom" else "Custom GPU",
    tflops=tflops,
    memory_bandwidth_GBs=memory_bandwidth_GBs,
    memory_size_GB=memory_size_GB,
    num_gpus=num_gpus,
    tensor_parallel=tensor_parallel,
    nvlink_bandwidth_GBs=nvlink_bandwidth_GBs,
    compute_utilization=compute_utilization,
    memory_utilization=memory_utilization
)

if model_choice == "Custom":
    model = ModelParams(
        name=model_name,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        dtype_bytes=dtype_bytes
    )
else:
    model = ModelParams(name=model_name, dtype_bytes=dtype_bytes)

# Create the tokenomics model
if use_advanced_model:
    tokenomics = AdvancedTokenomicsModel(model, hardware)
else:
    tokenomics = TokenomicsModel(model, hardware)

# Build base config for sweep analysis
base_config = {
    "hardware": {
        "num_gpus": num_gpus,
        "tensor_parallel": tensor_parallel,
        "compute_utilization": compute_utilization,
        "memory_utilization": memory_utilization,
        "tflops": tflops,
        "memory_bandwidth_GBs": memory_bandwidth_GBs,
        "memory_size_GB": memory_size_GB,
        "nvlink_bandwidth_GBs": nvlink_bandwidth_GBs
    },
    "model": {
        "name": model_name,
        "hidden_size": model.hidden_size,
        "num_hidden_layers": model.num_hidden_layers,
        "num_attention_heads": model.num_attention_heads,
        "num_key_value_heads": model.num_key_value_heads,
        "intermediate_size": model.intermediate_size,
        "vocab_size": model.vocab_size,
        "dtype_bytes": model.dtype_bytes
    },
    "inference": {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "batch_size": batch_size
    }
}

# === Check if we're in sweep mode ===
if st.session_state.sweep_variable is not None:
    sweep_var = st.session_state.sweep_variable
    config = SWEEP_CONFIGS[sweep_var]

    st.header(f"Sweep Analysis: {config['name']}")
    st.markdown(f"*Showing how all output metrics change as **{config['name']}** varies, with all other parameters held fixed.*")

    if st.button("Back to Dashboard", type="primary"):
        clear_sweep()
        st.rerun()

    st.divider()

    # Run the sweep
    with st.spinner(f"Running sweep analysis for {config['name']}..."):
        results = run_sweep_analysis(
            sweep_var, base_config, model_choice, gpu_choice, gpu_presets, use_advanced_model
        )

    if not results:
        st.error("No valid results from sweep. Check parameter ranges.")
    else:
        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Current value marker
        current_val = base_config["hardware"].get(config["param_key"]) or base_config["inference"].get(config["param_key"])

        # === Throughput Metrics ===
        st.subheader("Throughput Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Total Throughput (tokens/second)**")
            chart_data = df[["sweep_value", "total_throughput"]].rename(
                columns={"sweep_value": config["name"], "total_throughput": "Total Throughput"}
            )
            st.line_chart(chart_data.set_index(config["name"]))

        with col2:
            st.markdown("**Per-Request Throughput (tokens/second)**")
            chart_data = df[["sweep_value", "per_request_throughput"]].rename(
                columns={"sweep_value": config["name"], "per_request_throughput": "Per-Request Throughput"}
            )
            st.line_chart(chart_data.set_index(config["name"]))

        # === Timing Metrics ===
        st.subheader("Timing Metrics")

        # Add unified line chart for time breakdown
        st.markdown("**Time Breakdown: Prefill vs Decode**")

        # Prepare data for line chart - convert decode_time to ms for consistency
        breakdown_df = df[["sweep_value", "prefill_time_ms"]].copy()
        breakdown_df["decode_time_ms"] = df["decode_time"] * 1000  # Convert to ms
        breakdown_df = breakdown_df.rename(columns={
            "sweep_value": config["name"],
            "prefill_time_ms": "Prefill (ms)",
            "decode_time_ms": "Decode (ms)"
        })
        st.line_chart(breakdown_df.set_index(config["name"]))
        st.caption("Comparison of prefill and decode times across sweep values")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Total Inference Time (seconds)**")
            chart_data = df[["sweep_value", "total_time"]].rename(
                columns={"sweep_value": config["name"], "total_time": "Total Time (s)"}
            )
            st.line_chart(chart_data.set_index(config["name"]))

        with col4:
            st.markdown("**Avg Time per Token (ms)**")
            chart_data = df[["sweep_value", "avg_time_per_token_ms"]].rename(
                columns={"sweep_value": config["name"], "avg_time_per_token_ms": "Avg Time/Token (ms)"}
            )
            st.line_chart(chart_data.set_index(config["name"]))

        # For input_tokens sweep, add a log-log plot to show quadratic scaling
        if sweep_var == "input_tokens":
            import plotly.express as px
            import plotly.graph_objects as go

            st.subheader("Prefill Time Scaling Analysis (Log-Log)")
            st.markdown("*On a log-log plot, quadratic scaling (O(S²)) appears as a line with slope ≈ 2*")

            # Create log-log plot using plotly
            fig = go.Figure()

            # Add prefill time trace
            fig.add_trace(go.Scatter(
                x=df["sweep_value"],
                y=df["prefill_time_ms"],
                mode='lines+markers',
                name='Prefill Time (ms)',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))

            # Add a reference line for perfect quadratic scaling
            # If prefill time at first point is t0 at x0, quadratic would be t0 * (x/x0)^2
            x0 = df["sweep_value"].iloc[0]
            t0 = df["prefill_time_ms"].iloc[0]
            quadratic_ref = [t0 * (x / x0) ** 2 for x in df["sweep_value"]]

            fig.add_trace(go.Scatter(
                x=df["sweep_value"],
                y=quadratic_ref,
                mode='lines',
                name='Perfect O(S²) Reference',
                line=dict(color='red', width=2, dash='dash')
            ))

            fig.update_layout(
                xaxis_type="log",
                yaxis_type="log",
                xaxis_title="Input Tokens (log scale)",
                yaxis_title="Prefill Time in ms (log scale)",
                title="Prefill Time vs Input Tokens (Log-Log Scale)",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("The prefill time closely follows the O(S²) reference line, confirming quadratic scaling with sequence length.")

        # === Bottleneck Analysis ===
        st.subheader("Bottleneck Analysis")

        col7, col8 = st.columns(2)

        with col7:
            st.markdown("**Compute vs Memory Time (ms)**")
            chart_data = df[["sweep_value", "compute_time_ms", "memory_time_ms"]].rename(
                columns={"sweep_value": config["name"], "compute_time_ms": "Compute Time", "memory_time_ms": "Memory Time"}
            )
            st.line_chart(chart_data.set_index(config["name"]))

        with col8:
            st.markdown("**Compute/Memory Ratio**")
            # Cap ratio for visualization
            df_ratio = df.copy()
            df_ratio["compute_memory_ratio"] = df_ratio["compute_memory_ratio"].clip(upper=10)
            chart_data = df_ratio[["sweep_value", "compute_memory_ratio"]].rename(
                columns={"sweep_value": config["name"], "compute_memory_ratio": "Compute/Memory Ratio"}
            )
            st.line_chart(chart_data.set_index(config["name"]))
            st.caption("Ratio > 1 = Compute-bound, Ratio < 1 = Memory-bound")

        # === Memory Metrics ===
        st.subheader("Memory Metrics")

        col9, col10 = st.columns(2)

        with col9:
            st.markdown("**KV Cache Size (GB)**")
            chart_data = df[["sweep_value", "kv_cache_gb"]].rename(
                columns={"sweep_value": config["name"], "kv_cache_gb": "KV Cache (GB)"}
            )
            st.line_chart(chart_data.set_index(config["name"]))

        with col10:
            st.markdown("**Communication Overhead (ms)**")
            chart_data = df[["sweep_value", "comm_overhead_ms"]].rename(
                columns={"sweep_value": config["name"], "comm_overhead_ms": "Comm Overhead (ms)"}
            )
            st.line_chart(chart_data.set_index(config["name"]))

        # === Memory Load Breakdown ===
        st.subheader("Memory Load Breakdown (Decode Phase)")

        col11, col12 = st.columns(2)

        with col11:
            st.markdown("**Model Weights vs KV Cache Load Time (ms)**")
            chart_data = df[["sweep_value", "model_load_time_ms", "kv_cache_load_time_ms"]].rename(
                columns={"sweep_value": config["name"], "model_load_time_ms": "Model Weights", "kv_cache_load_time_ms": "KV Cache"}
            )
            st.line_chart(chart_data.set_index(config["name"]))

        with col12:
            st.markdown("**Model Size vs KV Cache Size (GB)**")
            chart_data = df[["sweep_value", "model_size_gb", "kv_cache_gb"]].rename(
                columns={"sweep_value": config["name"], "model_size_gb": "Model Weights", "kv_cache_gb": "KV Cache"}
            )
            st.line_chart(chart_data.set_index(config["name"]))

        # === Bottleneck Transitions ===
        st.subheader("Bottleneck Transitions")

        bottleneck_df = df[["sweep_value", "bottleneck"]].copy()
        bottleneck_df["is_memory_bound"] = (bottleneck_df["bottleneck"] == "memory").astype(int)

        # Find transition points
        transitions = []
        for i in range(1, len(bottleneck_df)):
            if bottleneck_df.iloc[i]["bottleneck"] != bottleneck_df.iloc[i-1]["bottleneck"]:
                transitions.append({
                    "from_value": bottleneck_df.iloc[i-1]["sweep_value"],
                    "to_value": bottleneck_df.iloc[i]["sweep_value"],
                    "from_bottleneck": bottleneck_df.iloc[i-1]["bottleneck"].upper(),
                    "to_bottleneck": bottleneck_df.iloc[i]["bottleneck"].upper()
                })

        if transitions:
            st.markdown("**Bottleneck transition points:**")
            for t in transitions:
                st.info(f"Transition from **{t['from_bottleneck']}** to **{t['to_bottleneck']}** between {config['name']} = {t['from_value']} and {t['to_value']}")
        else:
            current_bottleneck = bottleneck_df.iloc[0]["bottleneck"].upper()
            st.success(f"System remains **{current_bottleneck}**-bound across all tested values.")

        # === Data Table ===
        st.subheader("Full Sweep Data")

        display_df = df[[
            "sweep_value", "total_throughput", "per_request_throughput",
            "total_time", "prefill_time_ms", "decode_time", "avg_time_per_token_ms",
            "compute_time_ms", "memory_time_ms", "model_load_time_ms", "kv_cache_load_time_ms",
            "compute_memory_ratio", "kv_cache_gb", "bottleneck"
        ]].copy()

        display_df.columns = [
            config["name"], "Total Throughput (tok/s)", "Per-Request (tok/s)",
            "Total Time (s)", "Prefill (ms)", "Decode (s)", "Avg/Token (ms)",
            "Compute (ms)", "Memory (ms)", "Model Load (ms)", "KV Load (ms)",
            "Compute/Memory", "KV Cache (GB)", "Bottleneck"
        ]

        # Round numeric columns for display but keep as numeric for sorting
        numeric_cols = [col for col in display_df.columns if col not in [config["name"], "Bottleneck"]]
        for col in numeric_cols:
            display_df[col] = display_df[col].round(4)

        display_df["Bottleneck"] = display_df["Bottleneck"].str.upper()

        st.dataframe(display_df, use_container_width=True)

        # === Equations for this variable ===
        st.subheader("Relevant Equations")

        if sweep_var == "batch_size":
            st.markdown(r"""
            **Batch size affects throughput through:**

            $$\text{Total Throughput} = \frac{B \times N_{\text{output}}}{T_{\text{total}}}$$

            As batch size $B$ increases:
            - Total throughput increases (more tokens processed in parallel)
            - Per-request throughput typically decreases (more contention)
            - KV cache size scales linearly: $\text{KV Cache} \propto B$
            - System may transition from memory-bound to compute-bound
            """)
        elif sweep_var == "input_tokens":
            st.markdown(r"""
            **Input token count affects:**

            $$\text{Prefill FLOPs} \propto S^2$$ (quadratic with sequence length)

            $$\text{KV Cache} = 2 \times d \times L \times h \times n_{kv} \times S \times B$$

            As input tokens $S$ increase:
            - Prefill time grows quadratically
            - KV cache size grows linearly
            - Decode attention becomes more expensive (attending to longer context)
            """)
        elif sweep_var == "output_tokens":
            st.markdown(r"""
            **Output token count affects:**

            $$T_{\text{decode}} = \sum_{i=1}^{N_{\text{output}}} T_{\text{token}}(S + i)$$

            As output tokens increase:
            - Total decode time grows approximately linearly
            - Each subsequent token is slightly slower (longer context)
            - KV cache grows as tokens are generated
            """)
        elif sweep_var == "num_gpus":
            st.markdown(r"""
            **GPU count affects effective compute and bandwidth:**

            $$\text{Effective TFLOPs} = \text{TFLOPs}_{\text{per GPU}} \times N_{\text{GPU}} \times \eta_{\text{compute}}$$

            $$\text{Effective BW} = \text{BW}_{\text{per GPU}} \times N_{\text{GPU}} \times \eta_{\text{memory}}$$

            With tensor parallelism, adding GPUs:
            - Increases effective compute capacity
            - Increases effective memory bandwidth
            - Adds communication overhead between GPUs
            """)
        elif sweep_var in ["compute_utilization", "memory_utilization"]:
            st.markdown(r"""
            **Utilization rates scale effective hardware capacity:**

            $$\text{Effective TFLOPs} = \text{Peak TFLOPs} \times \eta_{\text{compute}}$$

            $$\text{Effective BW} = \text{Peak BW} \times \eta_{\text{memory}}$$

            Higher utilization means:
            - More of the theoretical peak is achieved
            - Real-world utilization is typically 50-80% due to overhead
            """)

else:
    # === Normal Dashboard View ===

    # Configuration Summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Hardware Summary")
        st.markdown(f"""
        - **GPU**: {hardware.name}
        - **GPUs**: {hardware.num_gpus}x
        - **Peak TFLOPs**: {hardware.tflops}
        - **Effective TFLOPs**: {hardware.effective_tflops:.1f}
        - **Memory BW**: {hardware.memory_bandwidth_GBs} GB/s
        - **Effective BW**: {hardware.effective_memory_bandwidth_GBs:.1f} GB/s
        - **Total VRAM**: {hardware.total_memory_GB} GB
        """)

    with col2:
        st.markdown("### Model Summary")
        st.markdown(f"""
        - **Model**: {model.name}
        - **Parameters**: {model.total_params / 1e9:.2f}B
        - **Model Size**: {model.model_size_bytes / 1e9:.2f} GB
        - **Hidden Size**: {model.hidden_size}
        - **Layers**: {model.num_hidden_layers}
        - **Attention Heads**: {model.num_attention_heads}
        - **KV Heads**: {model.num_key_value_heads}
        """)

    with col3:
        st.markdown("### Inference Settings")
        st.markdown(f"""
        - **Input Tokens**: {input_tokens}
        - **Output Tokens**: {output_tokens}
        - **Batch Size**: {batch_size}
        - **Total Context**: {input_tokens + output_tokens}
        - **Simulation**: {"Advanced" if use_advanced_model else "Basic"}
        """)

    st.divider()

    # === Compute Metrics ===
    st.header("Computed Metrics")

    metrics = compute_metrics(tokenomics, input_tokens, output_tokens, batch_size, hardware)

    # Display key metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Total Throughput", f"{metrics['total_throughput']:.1f} tok/s")
        st.metric("Per-Request Throughput", f"{metrics['per_request_throughput']:.1f} tok/s")

    with metric_col2:
        st.metric("Total Inference Time", f"{metrics['total_time']:.3f} s")
        st.metric("Prefill Time", f"{metrics['prefill_time_ms']:.2f} ms")

    with metric_col3:
        st.metric("Decode Time (Total)", f"{metrics['decode_time']:.3f} s")
        st.metric("Avg Time per Token", f"{metrics['avg_time_per_token_ms']:.2f} ms")

    with metric_col4:
        st.metric("Bottleneck", metrics['bottleneck'].upper())
        st.metric("KV Cache Size", f"{metrics['kv_cache_gb']:.2f} GB")

    st.divider()

    # === Detailed Breakdown ===
    st.header("Detailed Time Breakdown")

    breakdown = tokenomics.decode_time_breakdown(input_tokens + output_tokens, batch_size)

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.subheader("Decode Step Analysis")
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Compute Time | {breakdown['compute_time'] * 1000:.4f} ms |
        | Memory Time | {breakdown['memory_time'] * 1000:.4f} ms |
        | Comm Overhead | {breakdown['comm_overhead'] * 1000:.4f} ms |
        | Total (per token) | {breakdown['total_time'] * 1000:.4f} ms |
        | **Bottleneck** | **{breakdown['bottleneck'].upper()}** |
        """)

        # Memory load breakdown: model weights vs KV cache
        st.subheader("Memory Load Breakdown")
        total_load_time = metrics['model_load_time_ms'] + metrics['kv_cache_load_time_ms']
        model_load_pct = (metrics['model_load_time_ms'] / total_load_time * 100) if total_load_time > 0 else 100
        kv_load_pct = (metrics['kv_cache_load_time_ms'] / total_load_time * 100) if total_load_time > 0 else 0

        st.markdown(f"""
        | Component | Size | Load Time | % |
        |-----------|------|-----------|---|
        | Model Weights | {metrics['model_size_gb']:.2f} GB | {metrics['model_load_time_ms']:.4f} ms | {model_load_pct:.1f}% |
        | KV Cache | {metrics['kv_cache_gb']:.2f} GB | {metrics['kv_cache_load_time_ms']:.4f} ms | {kv_load_pct:.1f}% |
        | **Total** | {metrics['model_size_gb'] + metrics['kv_cache_gb']:.2f} GB | {total_load_time:.4f} ms | 100% |
        """)

        # Visual bar showing KV cache as proportion of memory load
        kv_bar_pct = kv_load_pct / 100
        st.progress(kv_bar_pct)

        if use_advanced_model:
            st.subheader("Advanced Model Factors")
            st.markdown(f"""
            | Factor | Value |
            |--------|-------|
            | KV Cache Penalty | {breakdown.get('kv_cache_penalty', 1.0):.3f}x |
            | Fragmentation Factor | {breakdown.get('fragmentation_factor', 1.0):.3f}x |
            | Effective Memory BW | {breakdown.get('memory_bandwidth_GBs', hardware.effective_memory_bandwidth_GBs):.1f} GB/s |
            | NCCL Overhead | {breakdown.get('nccl_overhead', 0) * 1000:.4f} ms |
            | Kernel Overhead | {breakdown.get('kernel_overhead', 0) * 1000:.4f} ms |
            """)

    with detail_col2:
        st.subheader("Compute/Memory Ratio")
        compute_memory_ratio = metrics['compute_memory_ratio']

        st.markdown(f"""
        **Compute/Memory Time Ratio**: `{compute_memory_ratio:.4f}`

        - If ratio < 1: **Memory-bound** (memory is the bottleneck)
        - If ratio > 1: **Compute-bound** (compute is the bottleneck)
        - If ratio = 1: **Balanced** (optimal utilization)
        """)

        # Visual indicator
        ratio_pct = min(compute_memory_ratio / 2, 1.0)
        st.progress(ratio_pct)

        if compute_memory_ratio < 0.5:
            st.warning("System is heavily memory-bound. Consider increasing batch size to better amortize memory access, or use GPUs with higher memory bandwidth.")
        elif compute_memory_ratio > 2.0:
            st.warning("System is heavily compute-bound. Memory bandwidth is underutilized. Consider reducing batch size for lower latency, or use GPUs with higher TFLOPs.")
        else:
            st.success("System is reasonably balanced.")

    st.divider()

    # === Equations Reference ===
    st.header("Equations Reference")

    # Pre-compute values for examples
    S = input_tokens  # sequence length
    H = model.hidden_size
    A = model.num_attention_heads
    V = model.vocab_size
    L = model.num_hidden_layers
    B = batch_size
    d_type = model.dtype_bytes
    h_size = model.head_size
    n_kv = model.num_key_value_heads

    # Computed values
    kv_cache_bytes = tokenomics.calculate_kv_cache_size(input_tokens, batch_size)
    kv_cache_gb = kv_cache_bytes / 1e9
    model_size_gb = model.model_size_bytes / 1e9
    eff_tflops = hardware.effective_tflops
    eff_bw = hardware.effective_memory_bandwidth_GBs

    # Get prefill and decode FLOPs
    prefill_flops = tokenomics.calculate_prefill_flops(input_tokens)
    prefill_flops_batch = prefill_flops * batch_size
    decode_flops = tokenomics.calculate_decode_flops(input_tokens)
    decode_flops_batch = decode_flops * batch_size

    # Calculate prefill memory (activations, not weights - weights are compute-bound in prefill)
    prefill_activation_bytes = batch_size * input_tokens * H * d_type
    prefill_activation_gb = prefill_activation_bytes / 1e9

    with st.expander("Overview: Inference Phases", expanded=True):
        st.markdown("""
        ### LLM Inference consists of two main phases:

        | Phase | Description | Parallelism | Bottleneck |
        |-------|-------------|-------------|------------|
        | **Prefill** | Process entire input prompt at once | High (all tokens in parallel) | Usually **Compute-bound** |
        | **Decode** | Generate one token at a time | Low (sequential) | Usually **Memory-bound** |

        Each phase has both compute and memory components:
        """)
        st.markdown(f"""
        ```
        Total Inference Time = Prefill Time + Decode Time

        Prefill Time = max(Prefill Compute, Prefill Memory) + Prefill Comm
        Decode Time  = Σ [max(Decode Compute, Decode Memory) + Decode Comm] for each output token

        Current Configuration:
        - Prefill Time: {metrics['prefill_time_ms']:.4f} ms
        - Decode Time:  {metrics['decode_time'] * 1000:.4f} ms (for {output_tokens} tokens)
        - Total Time:   {metrics['total_time'] * 1000:.4f} ms
        ```
        """)

    with st.expander("Prefill Phase"):
        st.markdown(r"""
        ## Compute

        ### Prefill Compute Time
        During prefill, all input tokens are processed in parallel through the transformer layers.

        $$
        T_{\text{prefill\_compute}} = \frac{\text{Prefill FLOPs}}{\text{Effective TFLOPs} \times 10^{12}}
        $$

        ### Prefill FLOPs Breakdown (per layer)

        **Attention FLOPs:**
        - RMS Norm: $4 \times S \times H$
        - Query Projection: $2 \times S \times H^2$
        - KV Projections: $0.5 \times S \times H^2$ (reduced for GQA)
        - RoPE: $6 \times S \times H$
        - Q @ K^T: $2 \times S^2 \times H$
        - Softmax: $5 \times S^2 \times A$
        - Attention Output: $2 \times S^2 \times H$
        - O-Projection: $2 \times S \times H^2$

        **MLP FLOPs:**
        - Gate + Up + Down projections: $21 \times S \times H^2$

        **LM Head:**
        - Final projection: $2 \times S \times H \times V$

        **Total:**
        $$
        \text{Prefill FLOPs} = (\text{Attention} + \text{MLP}) \times L + \text{LM Head}
        $$
        """)

        prefill_compute_time = (prefill_flops_batch / 1e12) / eff_tflops
        st.markdown(f"""
        **Example with current values:**
        ```
        Sequence Length (S) = {S}
        Hidden Size (H) = {H}
        Num Layers (L) = {L}
        Batch Size (B) = {B}

        Prefill FLOPs (single sequence) = {prefill_flops:,.0f}
        Prefill FLOPs (batch of {B}) = {prefill_flops_batch:,.0f}
                                     = {prefill_flops_batch / 1e12:.4f} TFLOPs

        Prefill Compute Time = {prefill_flops_batch / 1e12:.4f} TFLOPs / {eff_tflops:.1f} TFLOPs
                            = {prefill_compute_time * 1000:.4f} ms
        ```
        """)

        st.markdown("---")

        st.markdown(r"""
        ## Memory

        ### Prefill Memory Access
        During prefill, memory access includes:
        1. **Loading model weights** (each layer's weights loaded once)
        2. **Writing KV cache** (K and V tensors for all input tokens)
        3. **Activation memory** (intermediate tensors)

        For large batch sizes with long sequences, prefill is typically **compute-bound** because:
        - Model weights are reused across all tokens (high arithmetic intensity)
        - The O(S²) attention computation dominates

        $$
        T_{\text{prefill\_memory}} = \frac{\text{Model Size} + \text{Activation Memory}}{\text{Effective Bandwidth}}
        $$
        """)

        prefill_memory_time = (model_size_gb + prefill_activation_gb) / eff_bw
        st.markdown(f"""
        **Example with current values:**
        ```
        Model Size = {model_size_gb:.2f} GB
        Activation Memory ≈ B × S × H × dtype = {B} × {S} × {H} × {d_type}
                        = {prefill_activation_bytes:,.0f} bytes
                        = {prefill_activation_gb:.4f} GB

        Prefill Memory Time = ({model_size_gb:.2f} + {prefill_activation_gb:.4f}) GB / {eff_bw:.1f} GB/s
                           = {prefill_memory_time * 1000:.4f} ms
        ```
        """)

        st.markdown("---")

        st.markdown(f"""
        ## Summary
        ```
        Prefill Compute Time: {prefill_compute_time * 1000:.4f} ms
        Prefill Memory Time:  {prefill_memory_time * 1000:.4f} ms

        Prefill is {'COMPUTE' if prefill_compute_time > prefill_memory_time else 'MEMORY'}-BOUND
        ```
        """)

    with st.expander("Decode Phase"):
        st.markdown(r"""
        ## Compute

        ### Decode Compute Time (per token)
        During decode, only ONE new token is processed at a time.
        With KV caching, we only compute:
        - Projections for the new token (not cached tokens)
        - Attention between new query and ALL cached keys/values

        $$
        T_{\text{decode\_compute}} = \frac{\text{Decode FLOPs}}{\text{Effective TFLOPs} \times 10^{12}}
        $$

        ### Decode FLOPs Breakdown (per layer, per token)

        **Attention FLOPs:**
        - RMS Norm: $4 \times H$ (single token)
        - Query Projection: $2 \times H^2$
        - KV Projections: $0.5 \times H^2$
        - RoPE: $6 \times H$
        - Q @ cached K^T: $2 \times S \times H$ (S = current context length)
        - Softmax: $5 \times S \times A$
        - Attention Output: $2 \times S \times H$
        - O-Projection: $2 \times H^2$

        **MLP FLOPs:** $21 \times H^2$

        **LM Head:** $2 \times H \times V$
        """)

        decode_compute_time = (decode_flops_batch / 1e12) / eff_tflops
        st.markdown(f"""
        **Example with current values (first decode token):**
        ```
        Context Length (S) = {S} (grows by 1 each token)
        Hidden Size (H) = {H}

        Decode FLOPs (single sequence) = {decode_flops:,.0f}
        Decode FLOPs (batch of {B}) = {decode_flops_batch:,.0f}
                                    = {decode_flops_batch / 1e12:.6f} TFLOPs

        Decode Compute Time = {decode_flops_batch / 1e12:.6f} TFLOPs / {eff_tflops:.1f} TFLOPs
                           = {decode_compute_time * 1000:.4f} ms per token
        ```
        """)

        st.markdown("---")

        st.markdown(r"""
        ## Memory

        ### Decode Memory Time (per token)
        During decode, memory bandwidth is often the bottleneck because:
        1. **Model weights must be loaded** for each token (low reuse)
        2. **KV cache must be read** (grows with context length)
        3. Low arithmetic intensity (few FLOPs per byte loaded)

        $$
        T_{\text{decode\_memory}} = \frac{\text{Model Size} + \text{KV Cache Size}}{\text{Effective Bandwidth}}
        $$

        ### KV Cache Size
        $$
        \text{KV Cache} = 2 \times d_{\text{type}} \times L \times h_{\text{size}} \times n_{\text{kv}} \times S \times B
        $$
        """)

        decode_memory_time = (model_size_gb + kv_cache_gb) / eff_bw
        st.markdown(f"""
        **Example with current values:**
        ```
        Model Size = {model_size_gb:.2f} GB (loaded every token!)

        KV Cache = 2 × {d_type} × {L} × {h_size} × {n_kv} × {S} × {B}
                = {kv_cache_bytes:,.0f} bytes
                = {kv_cache_gb:.4f} GB

        Total Memory to Load = {model_size_gb:.2f} + {kv_cache_gb:.4f} = {model_size_gb + kv_cache_gb:.4f} GB

        Decode Memory Time = {model_size_gb + kv_cache_gb:.4f} GB / {eff_bw:.1f} GB/s
                          = {decode_memory_time * 1000:.4f} ms per token

        Memory Breakdown:
        - Model Weights: {model_size_gb:.2f} GB / {eff_bw:.1f} GB/s = {(model_size_gb / eff_bw) * 1000:.4f} ms ({model_size_gb / (model_size_gb + kv_cache_gb) * 100:.1f}%)
        - KV Cache:      {kv_cache_gb:.4f} GB / {eff_bw:.1f} GB/s = {(kv_cache_gb / eff_bw) * 1000:.4f} ms ({kv_cache_gb / (model_size_gb + kv_cache_gb) * 100:.1f}%)
        ```
        """)

        st.markdown("---")

        st.markdown(r"""
        ## Bottleneck Analysis

        The decode phase is bottlenecked by whichever takes longer:

        $$
        T_{\text{decode\_token}} = \max(T_{\text{compute}}, T_{\text{memory}}) + T_{\text{comm}}
        $$

        $$
        \text{Compute/Memory Ratio} = \frac{T_{\text{compute}}}{T_{\text{memory}}}
        $$

        - Ratio < 1: **Memory-bound** (GPU waiting for data)
        - Ratio > 1: **Compute-bound** (GPU busy computing)
        - Ratio ≈ 1: **Balanced** (optimal utilization)
        """)

        st.markdown(f"""
        **Example with current values:**
        ```
        Decode Compute Time: {metrics['compute_time_ms']:.4f} ms
        Decode Memory Time:  {metrics['memory_time_ms']:.4f} ms

        Compute/Memory Ratio = {metrics['compute_time_ms']:.4f} / {metrics['memory_time_ms']:.4f}
                            = {metrics['compute_memory_ratio']:.4f}

        {metrics['compute_time_ms']:.4f} {'<' if metrics['compute_time_ms'] < metrics['memory_time_ms'] else '>'} {metrics['memory_time_ms']:.4f}
        → System is {metrics['bottleneck'].upper()}-BOUND

        Decode Time per Token = max({metrics['compute_time_ms']:.4f}, {metrics['memory_time_ms']:.4f}) + {breakdown.get('comm_overhead', 0) * 1000:.4f}
                             = {breakdown['total_time'] * 1000:.4f} ms
        ```
        """)

    with st.expander("Communication Overhead"):
        comm_overhead_prefill = breakdown.get('comm_overhead', 0) * 1000 * input_tokens  # rough estimate
        comm_overhead_decode = breakdown.get('comm_overhead', 0) * 1000

        st.markdown(r"""
        ### Tensor Parallel Communication
        When using multiple GPUs with tensor parallelism, each layer requires:
        - **All-reduce** after attention
        - **All-reduce** after MLP

        $$
        T_{\text{comm}} = \text{Fixed Latency} + \frac{\text{Data Size}}{\text{NVLink Bandwidth}}
        $$

        **Prefill Communication:**
        - Data size per communication: $B \times S \times H \times d_{type}$
        - Number of communications: $2 \times L$

        **Decode Communication:**
        - Data size per communication: $B \times H \times d_{type}$
        - Number of communications: $2 \times L$
        """)

        if hardware.tensor_parallel and hardware.num_gpus > 1:
            prefill_comm_size = batch_size * input_tokens * H * d_type
            decode_comm_size = batch_size * H * d_type
            nvlink_bw = hardware.nvlink_bandwidth_GBs if hardware.nvlink_bandwidth_GBs > 0 else 32.0

            st.markdown(f"""
            **Example with current values (Tensor Parallel across {hardware.num_gpus} GPUs):**
            ```
            NVLink Bandwidth = {nvlink_bw} GB/s

            Prefill Communication:
            - Data per comm = {B} × {S} × {H} × {d_type} = {prefill_comm_size:,} bytes = {prefill_comm_size / 1e9:.6f} GB
            - Total comms = 2 × {L} = {2 * L}
            - Comm overhead ≈ {2 * L} × {prefill_comm_size / 1e9:.6f} GB / {nvlink_bw} GB/s = varies

            Decode Communication:
            - Data per comm = {B} × {H} × {d_type} = {decode_comm_size:,} bytes = {decode_comm_size / 1e9:.6f} GB
            - Total comms = 2 × {L} = {2 * L}
            - Comm overhead = {breakdown.get('comm_overhead', 0) * 1000:.4f} ms per token
            ```
            """)
        else:
            st.markdown("""
            **Current setting:** Single GPU or tensor parallelism disabled → No communication overhead
            """)

    with st.expander("Throughput Calculations"):
        st.markdown(r"""
        ### Total Throughput
        $$
        \text{Total Throughput} = \frac{\text{batch\_size} \times \text{output\_tokens}}{\text{total\_inference\_time}} \quad \text{[tokens/second]}
        $$

        ### Per-Request Throughput
        $$
        \text{Per-Request Throughput} = \frac{\text{Total Throughput}}{\text{batch\_size}} \quad \text{[tokens/second]}
        $$
        """)
        st.markdown(f"""
        **Example with current values:**
        ```
        Total Throughput = (batch_size × output_tokens) / total_time
                        = ({batch_size} × {output_tokens}) / {metrics['total_time']:.4f} s
                        = {batch_size * output_tokens} / {metrics['total_time']:.4f}
                        = {metrics['total_throughput']:.2f} tokens/second

        Per-Request Throughput = {metrics['total_throughput']:.2f} / {batch_size}
                              = {metrics['per_request_throughput']:.2f} tokens/second
        ```
        """)

    with st.expander("Hardware Efficiency Calculations"):
        st.markdown(r"""
        ### Effective TFLOPs
        $$
        \text{Effective TFLOPs} = \text{Peak TFLOPs} \times N_{\text{GPU}} \times \eta_{\text{compute}}
        $$

        ### Effective Memory Bandwidth
        $$
        \text{Effective BW} = \text{Peak BW} \times N_{\text{GPU}} \times \eta_{\text{memory}}
        $$
        """)
        st.markdown(f"""
        **Example with current values:**
        ```
        Effective TFLOPs = {hardware.tflops} × {hardware.num_gpus} × {hardware.compute_utilization}
                        = {eff_tflops:.1f} TFLOPs

        Effective BW = {hardware.memory_bandwidth_GBs} × {hardware.num_gpus} × {hardware.memory_utilization}
                    = {eff_bw:.1f} GB/s

        Arithmetic Intensity (Decode) = FLOPs / Bytes Loaded
                                     = {decode_flops_batch:,.0f} / {(model_size_gb + kv_cache_gb) * 1e9:,.0f}
                                     = {decode_flops_batch / ((model_size_gb + kv_cache_gb) * 1e9):.2f} FLOPs/byte
        ```

        For reference, the "roofline" crossover point is:
        ```
        Crossover = Effective TFLOPs / Effective BW
                 = {eff_tflops * 1e12:.0f} FLOPs/s / {eff_bw * 1e9:.0f} bytes/s
                 = {eff_tflops * 1e12 / (eff_bw * 1e9):.1f} FLOPs/byte

        Current arithmetic intensity ({decode_flops_batch / ((model_size_gb + kv_cache_gb) * 1e9):.2f}) {'<' if decode_flops_batch / ((model_size_gb + kv_cache_gb) * 1e9) < eff_tflops * 1e12 / (eff_bw * 1e9) else '>'} crossover ({eff_tflops * 1e12 / (eff_bw * 1e9):.1f})
        → Decode is {'MEMORY' if decode_flops_batch / ((model_size_gb + kv_cache_gb) * 1e9) < eff_tflops * 1e12 / (eff_bw * 1e9) else 'COMPUTE'}-BOUND
        ```
        """)

    if use_advanced_model:
        with st.expander("Advanced Model Factors"):
            # Get advanced model factors from breakdown
            kv_penalty = breakdown.get('kv_cache_penalty', 1.0)
            frag_factor = breakdown.get('fragmentation_factor', 1.0)
            adv_eff_bw = breakdown.get('memory_bandwidth_GBs', eff_bw)

            st.markdown(r"""
            ### KV Cache Access Penalty
            Models cache thrashing and TLB misses for large caches:
            $$
            \text{penalty} = 1.0 + 0.015 \times (\text{cache\_size\_GB})^{1.2}
            $$
            Capped at 2.0 maximum.
            """)
            st.markdown(f"""
            **Example with current values:**
            ```
            KV Cache Size = {kv_cache_gb:.4f} GB
            Penalty = 1.0 + 0.015 × ({kv_cache_gb:.4f})^1.2
                   = 1.0 + 0.015 × {kv_cache_gb**1.2:.4f}
                   = {kv_penalty:.4f}
            ```
            """)

            st.markdown(r"""
            ### Memory Fragmentation Factor
            $$
            \text{fragmentation} =
            \begin{cases}
            1.0 & \text{if } S \leq 8192 \\
            1.0 + 0.008 \times \frac{S - 8192}{1024} & \text{otherwise}
            \end{cases}
            $$
            Capped at 1.4 maximum.
            """)
            if input_tokens <= 8192:
                st.markdown(f"""
                **Example with current values:**
                ```
                Sequence Length = {input_tokens} ≤ 8192
                Fragmentation Factor = 1.0 (no fragmentation penalty)
                ```
                """)
            else:
                st.markdown(f"""
                **Example with current values:**
                ```
                Sequence Length = {input_tokens} > 8192
                Fragmentation = 1.0 + 0.008 × ({input_tokens} - 8192) / 1024
                             = 1.0 + 0.008 × {(input_tokens - 8192) / 1024:.2f}
                             = {frag_factor:.4f}
                ```
                """)

            st.markdown(r"""
            ### Effective Memory Bandwidth Reduction
            For sequences > 4096 tokens:
            $$
            \text{efficiency} = 1.0 - 0.05 \times \log_2\left(\frac{S}{4096}\right)
            $$

            $$
            \text{Adjusted BW} = \text{Base BW} \times 1.3 \times \max(0.75, \text{efficiency})
            $$
            """)
            import math
            if input_tokens <= 4096:
                st.markdown(f"""
                **Example with current values:**
                ```
                Sequence Length = {input_tokens} ≤ 4096
                Adjusted BW = Base BW × 1.3 (no efficiency reduction)
                           = {eff_bw:.1f} × 1.3
                           = {adv_eff_bw:.1f} GB/s
                ```
                """)
            else:
                efficiency = 1.0 - 0.05 * math.log2(input_tokens / 4096)
                efficiency = max(0.75, efficiency)
                st.markdown(f"""
                **Example with current values:**
                ```
                Sequence Length = {input_tokens} > 4096
                Efficiency = 1.0 - 0.05 × log2({input_tokens} / 4096)
                          = 1.0 - 0.05 × {math.log2(input_tokens / 4096):.2f}
                          = {efficiency:.4f}
                Adjusted BW = {eff_bw:.1f} × 1.3 × {efficiency:.4f}
                           = {adv_eff_bw:.1f} GB/s
                ```
                """)

            st.markdown(r"""
            ### Attention Algorithm Selection
            - **Standard Attention**: $S \leq 512$
            - **Flash Attention**: $512 < S \leq 8192$ (30-50% FLOP reduction)
            - **Block-Sparse Attention**: $S > 8192$ (70-80% FLOP reduction)
            """)
            if input_tokens <= 512:
                algo = "Standard Attention"
            elif input_tokens <= 8192:
                algo = "Flash Attention (30-50% FLOP reduction)"
            else:
                algo = "Block-Sparse Attention (70-80% FLOP reduction)"
            st.markdown(f"""
            **Current setting:** Sequence length = {input_tokens} → **{algo}**
            """)

            st.markdown(r"""
            ### Batch Efficiency Factor
            $$
            \eta_{\text{batch}} =
            \begin{cases}
            1.0 & B = 1 \\
            0.95 & B \leq 4 \\
            0.92 & B \leq 16 \\
            0.90 & B > 16
            \end{cases}
            $$
            """)
            if batch_size == 1:
                batch_eff = 1.0
            elif batch_size <= 4:
                batch_eff = 0.95
            elif batch_size <= 16:
                batch_eff = 0.92
            else:
                batch_eff = 0.90
            st.markdown(f"""
            **Current setting:** Batch size = {batch_size} → Efficiency = **{batch_eff}**
            """)

    st.divider()

    # === Batch Size Sweep ===
    st.header("Batch Size Sweep Analysis")
    st.markdown("*How metrics change across all batch sizes with current settings*")

    batch_sizes_sweep = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # Calculate which batch sizes would exceed memory (for warning purposes)
    model_size_gb = model.model_size_bytes / 1e9
    exceeds_memory = []
    for b in batch_sizes_sweep:
        total_mem = model_size_gb + tokenomics.calculate_kv_cache_size(input_tokens + output_tokens, b) / 1e9
        if total_mem > hardware.total_memory_GB:
            exceeds_memory.append(b)

    # Compute metrics for all batch sizes
    sweep_results = []
    for bs in batch_sizes_sweep:
        try:
            m = compute_metrics(tokenomics, input_tokens, output_tokens, bs, hardware)
            m["batch_size"] = bs
            # Check if this batch size would cause OOM
            total_mem = model_size_gb + tokenomics.calculate_kv_cache_size(input_tokens + output_tokens, bs) / 1e9
            m["oom"] = "Yes" if total_mem > hardware.total_memory_GB else "No"
            m["total_memory_gb"] = total_mem
            sweep_results.append(m)
        except Exception:
            pass

    if sweep_results:
        sweep_df = pd.DataFrame(sweep_results)

        # === Bottleneck Transitions ===
        transitions = []
        for i in range(1, len(sweep_df)):
            if sweep_df.iloc[i]["bottleneck"] != sweep_df.iloc[i-1]["bottleneck"]:
                transitions.append({
                    "from_bs": sweep_df.iloc[i-1]["batch_size"],
                    "to_bs": sweep_df.iloc[i]["batch_size"],
                    "from_bn": sweep_df.iloc[i-1]["bottleneck"].upper(),
                    "to_bn": sweep_df.iloc[i]["bottleneck"].upper()
                })

        if transitions:
            for t in transitions:
                st.info(f"Transition from **{t['from_bn']}** to **{t['to_bn']}** between batch size {t['from_bs']} and {t['to_bs']}")
        else:
            current_bn = sweep_df.iloc[0]["bottleneck"].upper()
            st.success(f"System remains **{current_bn}**-bound across all batch sizes.")

        # === Data Table ===
        display_df = sweep_df[[
            "batch_size", "total_throughput", "per_request_throughput",
            "total_time", "avg_time_per_token_ms",
            "compute_time_ms", "memory_time_ms", "model_load_time_ms", "kv_cache_load_time_ms",
            "compute_memory_ratio", "kv_cache_gb", "total_memory_gb", "bottleneck", "oom"
        ]].copy()

        display_df.columns = [
            "Batch Size", "Total Throughput (tok/s)", "Per-Request (tok/s)",
            "Total Time (s)", "Avg/Token (ms)",
            "Compute (ms)", "Memory (ms)", "Model Load (ms)", "KV Load (ms)",
            "Compute/Memory", "KV Cache (GB)", "Total Memory (GB)", "Bottleneck", "OOM"
        ]

        # Round numeric columns for display but keep as numeric for sorting
        numeric_cols = [col for col in display_df.columns if col not in ["Batch Size", "Bottleneck", "OOM"]]
        for col in numeric_cols:
            display_df[col] = display_df[col].round(4)

        display_df["Bottleneck"] = display_df["Bottleneck"].str.upper()
        st.dataframe(display_df, use_container_width=True)

        # Show OOM warning below the table
        if exceeds_memory:
            st.warning(f"Batch sizes {exceeds_memory} would exceed total GPU memory ({hardware.total_memory_GB} GB). Results shown for theoretical analysis.")

    st.divider()

    # Footer
    st.markdown("---")
    st.markdown("""
    *Dashboard based on the LLM Inference Economics simulation library.
    Equations derived from transformer architecture analysis with Llama-style optimizations (GQA, RoPE, SwiGLU).*

    **Tip:** Click the "Sweep" button next to any slider to see how that variable affects all output metrics.

    ---
    *Inspired by [LLM Inference Economics: Cost-Based Analysis and Future Trends](https://substack.com/home/post/p-163319195)*
    """)
