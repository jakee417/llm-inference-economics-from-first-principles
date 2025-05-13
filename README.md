# LLM inference economics from first principles

Codebase behind the "LLM inference economics from first principles" project, which models and analyzes LLM inference performance with a focus on throughput prediction and bottleneck analysis.

## Overview

This project provides tools to:
- Model theoretical throughput of LLM inference
- Compare theoretical vs. actual performance
- Analyze compute vs. memory bottlenecks
- Study how context length impacts performance
- Examine batch size scaling effects

## Installation

```
pip install -r requirements.txt
```

## File Structure

- `model_params.py`: Parameters for LLM model architectures
- `hardware_params.py`: Parameters for hardware specifications
- `tokenomics_model.py`: Basic throughput modeling
- `advanced_tokenomics_model.py`: Advanced model with realistic memory/attention effects
- `visualizations.py`: Plotting functions
- `main.py`: Command-line interface

## Models Supported

- **llama-3-1-8b**: Llama 3.1 8B parameter model
- **llama-3-3-70b**: Llama 3.3 70B parameter model

## Usage Examples

### Basic Theoretical Throughput

To generate a theoretical throughput estimate:

```bash
python main.py
```

### Comparing with Benchmark Data

Compare theoretical predictions with actual benchmark results:

```bash
python main.py --benchmark ../experiments/llama_33_70b_2000in_300out.json
```

### Using the Advanced Model

The advanced model incorporates realistic factors like memory fragmentation, attention optimizations, and more:

```bash
python main.py --benchmark ../experiments/llama_33_70b_16000in_1000out.json --advanced
```

### Specifying Different Model & Hardware Configurations

For Llama 3.1 8B on a single GPU:

```bash
python main.py --model llama-3-1-8b --gpus 1 -b ../experiments/llama_31_8b_2000in_300out.json --advanced
```

### Analyzing Long Context Performance

For long context lengths:

```bash
python main.py --model llama-3-1-8b --gpus 1 -b ../experiments/llama_31_8b_16000in_1000out.json --advanced
```

## Command Line Options

```
--benchmark, -b        Path to benchmark data JSON file
--model, -m            Model type (llama-3-1-8b or llama-3-3-70b)
--input-tokens, -i     Number of input tokens (default: 2035)
--output-tokens, -o    Number of output tokens (default: 300)
--gpus, -g             Number of GPUs (default: 4)
--advanced, -a         Use advanced model with realistic factors
--context-analysis, -c Generate context length scaling analysis
```

## Benchmark File Format

The benchmark files should be JSON with the following structure:

```json
{
  "metadata": {
    "model": "llama-3-3-70b",
    "batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
  },
  "results": {
    "1": {
      "avg_input_tokens": 2035,
      "avg_output_tokens": 300,
      "tokens_per_second_in_batch": 102.9,
      "avg_tokens_per_second": 102.9
    },
    "2": {
      "tokens_per_second_in_batch": 182.5,
      "avg_tokens_per_second": 91.25
    },
    ...
  }
}
```

## Understanding the Outputs

### Throughput Plots
The throughput plots show how many tokens per second the model can generate at different batch sizes:
- **Total Throughput**: Combined tokens/second across all batch entries
- **Per-Request Throughput**: Tokens/second for each individual request

### Bottleneck Analysis
Shows whether compute or memory bandwidth is the limiting factor at each batch size:
- **Blue markers**: Compute-bound regions
- **Red markers**: Memory-bound regions

### Context Length Analysis
When using the advanced model with `--context-analysis`, visualizes:
- How throughput decreases with context length
- Memory bandwidth efficiency changes
- KV cache penalties
- Memory fragmentation effects
