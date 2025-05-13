import matplotlib.pyplot as plt
import json

def plot_throughput_vs_batch_size(model, input_tokens=1024, output_tokens=128):
    """Plot total and per-request throughput against batch size with bottleneck analysis"""
    # Hardcoded batch sizes as requested
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    total_throughputs = []
    per_request_throughputs = []
    
    # Analyze bottlenecks
    bottleneck_analysis = model.analyze_bottlenecks(input_tokens, output_tokens, batch_sizes)
    compute_times = bottleneck_analysis["compute_times"]
    memory_times = bottleneck_analysis["memory_times"]
    bottlenecks = bottleneck_analysis["bottlenecks"]
    
    for batch_size in batch_sizes:
        total_throughput = model.calculate_total_throughput(input_tokens, output_tokens, batch_size)
        per_request_throughput = model.calculate_per_request_throughput(input_tokens, output_tokens, batch_size)
        
        total_throughputs.append(total_throughput)
        per_request_throughputs.append(per_request_throughput)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot total throughput
    ax1.plot(batch_sizes, total_throughputs, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Total Throughput (tokens/second)')
    ax1.set_title(f'Theoretical Total Throughput vs Batch Size\n{model.model.name} on {model.hardware.num_gpus}x {model.hardware.name}\nInput Tokens: {input_tokens}, Output Tokens: {output_tokens}')
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
    ax2.set_title(f'Theoretical Per-Request Throughput vs Batch Size\n{model.model.name} on {model.hardware.num_gpus}x {model.hardware.name}\nInput Tokens: {input_tokens}, Output Tokens: {output_tokens}')
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
    ax3.set_title(f'Compute vs Memory Bottleneck Analysis\n{model.model.name} on {model.hardware.num_gpus}x {model.hardware.name}\nInput Tokens: {input_tokens}, Output Tokens: {output_tokens}')
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
    fig.savefig(f"throughput_plots_{model.model.name.replace(' ', '_')}_{model.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png", 
                dpi=300, bbox_inches='tight')
    fig2.savefig(f"bottleneck_analysis_{model.model.name.replace(' ', '_')}_{model.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png", 
                dpi=300, bbox_inches='tight')
    
    print(f"\nFigures saved as:")
    print(f"- throughput_plots_{model.model.name.replace(' ', '_')}_{model.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png")
    print(f"- bottleneck_analysis_{model.model.name.replace(' ', '_')}_{model.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png")
    
    return fig, fig2, data

def plot_throughput_comparison(model, input_tokens, output_tokens, benchmark_data, batch_sizes=None):
    """
    Plot theoretical vs actual throughput against batch size with bottleneck analysis
    
    Args:
        model: TokenomicsModel instance
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        benchmark_data: Dictionary with batch sizes as keys and benchmark results as values
        batch_sizes: Optional list of batch sizes to use. If None, uses the keys from benchmark_data
    
    Returns:
        tuple: (throughput_plot, bottleneck_plot, efficiency_plot, data_points)
    """
    # Parse benchmark_data if it's a JSON string or file path
    if isinstance(benchmark_data, str):
        try:
            # Try to parse as JSON string
            results = json.loads(benchmark_data)
        except json.JSONDecodeError:
            # If not a valid JSON string, assume it's a file path
            with open(benchmark_data, 'r') as f:
                results = json.load(f)
            # Check if results has a "results" key (common format for benchmark files)
            if "results" in results:
                results = results["results"]
    else:
        # Already a dictionary
        results = benchmark_data
        # Check if results has a "results" key
        if "results" in results:
            results = results["results"]
    
    if batch_sizes is None:
        # Use the batch sizes from the benchmark results
        batch_sizes = sorted([int(k) for k in results.keys()])
    
    # Calculate theoretical throughputs
    theoretical_total_throughputs = []
    theoretical_per_request_throughputs = []
    
    # Analyze bottlenecks
    bottleneck_analysis = model.analyze_bottlenecks(input_tokens, output_tokens, batch_sizes)
    compute_times = bottleneck_analysis["compute_times"]
    memory_times = bottleneck_analysis["memory_times"]
    bottlenecks = bottleneck_analysis["bottlenecks"]
    
    # Get actual throughputs from benchmark data
    actual_total_throughputs = []
    actual_per_request_throughputs = []
    
    for batch_size in batch_sizes:
        # Theoretical throughputs
        total_throughput = model.calculate_total_throughput(input_tokens, output_tokens, batch_size)
        per_request_throughput = model.calculate_per_request_throughput(input_tokens, output_tokens, batch_size)
        
        theoretical_total_throughputs.append(total_throughput)
        theoretical_per_request_throughputs.append(per_request_throughput)
        
        # Actual throughputs (convert batch_size to string for dictionary lookup)
        bs_key = str(batch_size)
        if bs_key in results:
            actual_total_throughputs.append(results[bs_key].get("tokens_per_second_in_batch", None))
            actual_per_request_throughputs.append(results[bs_key].get("avg_tokens_per_second", None))
        else:
            actual_total_throughputs.append(None)
            actual_per_request_throughputs.append(None)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot total throughput (theoretical and actual)
    ax1.plot(batch_sizes, theoretical_total_throughputs, 'bo-', linewidth=2, markersize=8, label='Theoretical')
    ax1.plot(batch_sizes, actual_total_throughputs, 'go-', linewidth=2, markersize=8, label='Actual')
    ax1.set_xlabel('Batch Size', fontsize=14)
    ax1.set_ylabel('Total Throughput (tokens/second)', fontsize=14)
    ax1.set_title(f'Total Throughput vs Batch Size\n{model.model.name} on {model.hardware.num_gpus}x {model.hardware.name}\nInput Tokens: {input_tokens}, Output Tokens: {output_tokens}', 
                 fontsize=16, pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)  # Use log scale for x-axis
    ax1.set_xticks(batch_sizes)
    ax1.set_xticklabels(batch_sizes, fontsize=12)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis='y', labelsize=12)
    
    # Add values above points in total throughput plot for non-None values
    valid_theoretical = [x for x in theoretical_total_throughputs if x is not None]
    max_theoretical = max(valid_theoretical) if valid_theoretical else 0
    for i, v in enumerate(theoretical_total_throughputs):
        if v is not None:
            ax1.text(batch_sizes[i], float(v) + (float(max_theoretical) * 0.03), f"{float(v):.1f}", 
                    ha='center', va='bottom', fontweight='bold', color='blue', fontsize=12)
    
    valid_actual = [x for x in actual_total_throughputs if x is not None]
    max_actual = max(valid_actual) if valid_actual else 0
    for i, v in enumerate(actual_total_throughputs):
        if v is not None:
            ax1.text(batch_sizes[i], float(v) - (float(max_theoretical) * 0.07), f"{float(v):.1f}", 
                    ha='center', va='bottom', fontweight='bold', color='green', fontsize=12)
    
    # Plot per-request throughput (theoretical and actual)
    ax2.plot(batch_sizes, theoretical_per_request_throughputs, 'ro-', linewidth=2, markersize=8, label='Theoretical')
    ax2.plot(batch_sizes, actual_per_request_throughputs, 'mo-', linewidth=2, markersize=8, label='Actual')
    ax2.set_xlabel('Batch Size', fontsize=14)
    ax2.set_ylabel('Per-Request Throughput (tokens/second)', fontsize=14)
    ax2.set_title(f'Per-Request Throughput vs Batch Size\n{model.model.name} on {model.hardware.num_gpus}x {model.hardware.name}\nInput Tokens: {input_tokens}, Output Tokens: {output_tokens}', 
                 fontsize=16, pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)  # Use log scale for x-axis
    ax2.set_xticks(batch_sizes)
    ax2.set_xticklabels(batch_sizes, fontsize=12)
    ax2.legend(fontsize=14)
    ax2.tick_params(axis='y', labelsize=12)
    
    # Add values above points in per-request throughput plot for non-None values
    valid_theoretical_per = [x for x in theoretical_per_request_throughputs if x is not None]
    max_theoretical_per = max(valid_theoretical_per) if valid_theoretical_per else 0
    for i, v in enumerate(theoretical_per_request_throughputs):
        if v is not None:
            ax2.text(batch_sizes[i], float(v) + (float(max_theoretical_per) * 0.03), f"{float(v):.1f}", 
                    ha='center', va='bottom', fontweight='bold', color='red', fontsize=12)
    
    valid_actual_per = [x for x in actual_per_request_throughputs if x is not None]
    max_actual_per = max(valid_actual_per) if valid_actual_per else 0
    for i, v in enumerate(actual_per_request_throughputs):
        if v is not None:
            ax2.text(batch_sizes[i], float(v) - (float(max_theoretical_per) * 0.07), f"{float(v):.1f}", 
                    ha='center', va='bottom', fontweight='bold', color='magenta', fontsize=12)
    
    # Create a figure for bottleneck analysis
    fig2, ax3 = plt.subplots(figsize=(16, 10))
    
    # Plot compute and memory times
    ax3.plot(batch_sizes, compute_times, 'b-', linewidth=2, label='Compute Time')
    ax3.plot(batch_sizes, memory_times, 'r-', linewidth=2, label='Memory Time')
    
    # Add markers for bottlenecks
    for i, bottleneck in enumerate(bottlenecks):
        marker_color = 'blue' if bottleneck == 'compute' else 'red'
        marker_style = 'o' if bottleneck == 'compute' else 's'
        ax3.plot(batch_sizes[i], max(compute_times[i], memory_times[i]), 
                marker=marker_style, markersize=10, color=marker_color)
        
        # Add values above the bottleneck points
        max_time = max(compute_times[i], memory_times[i])
        ax3.text(batch_sizes[i], max_time + 0.0002, f"{max_time:.6f}",
                ha='center', va='bottom', fontweight='bold', color=marker_color, fontsize=10)
    
    ax3.set_xlabel('Batch Size', fontsize=14)
    ax3.set_ylabel('Time (seconds)', fontsize=14)
    ax3.set_title(f'Compute vs Memory Bottleneck Analysis\n{model.model.name} on {model.hardware.num_gpus}x {model.hardware.name}\nInput Tokens: {input_tokens}, Output Tokens: {output_tokens}', 
                 fontsize=16, pad=20)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)  # Use log scale for x-axis
    ax3.set_xticks(batch_sizes)
    ax3.set_xticklabels(batch_sizes, fontsize=12)
    ax3.legend(fontsize=14)
    ax3.tick_params(axis='y', labelsize=12)
    
    # Find crossover point
    for i in range(len(batch_sizes) - 1):
        if bottlenecks[i] != bottlenecks[i+1]:
            crossover_point = batch_sizes[i+1]
            ax3.axvline(x=crossover_point, color='green', linestyle='--', 
                    label=f'Bottleneck Shift at Batch Size {crossover_point}')
            ax3.legend()
            break
    
    # Create a third figure for the relative efficiency
    fig3, ax4 = plt.subplots(figsize=(16, 10))
    
    # Calculate efficiency (actual / theoretical)
    efficiencies = []
    for i, batch_size in enumerate(batch_sizes):
        if actual_total_throughputs[i] is None or theoretical_total_throughputs[i] == 0:
            efficiencies.append(None)
        else:
            efficiency = float(actual_total_throughputs[i]) / float(theoretical_total_throughputs[i]) * 100
            efficiencies.append(efficiency)
    
    # Plot efficiency
    ax4.plot(batch_sizes, efficiencies, 'ko-', linewidth=2, markersize=8)
    ax4.set_xlabel('Batch Size', fontsize=14)
    ax4.set_ylabel('Efficiency (Actual / Theoretical) %', fontsize=14)
    ax4.set_title(f'Inference Efficiency vs Batch Size\n{model.model.name} on {model.hardware.num_gpus}x {model.hardware.name}\nInput Tokens: {input_tokens}, Output Tokens: {output_tokens}', 
                 fontsize=16, pad=20)
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log', base=2)  # Use log scale for x-axis
    ax4.set_xticks(batch_sizes)
    ax4.set_xticklabels(batch_sizes, fontsize=12)
    ax4.tick_params(axis='y', labelsize=12)
    
    # Add values above points in efficiency plot for non-None values
    for i, v in enumerate(efficiencies):
        if v is not None:
            ax4.text(batch_sizes[i], v + 2, f"{v:.1f}%", 
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Create a data table with the values and bottlenecks
    data = []
    for i, bs in enumerate(batch_sizes):
        bs_key = str(bs)
        actual_total = None
        actual_per_request = None
        efficiency = None
        
        if bs_key in results:
            actual_total = results[bs_key].get("tokens_per_second_in_batch")
            actual_per_request = results[bs_key].get("avg_tokens_per_second")
            if actual_total is not None and theoretical_total_throughputs[i] > 0:
                efficiency = (actual_total / theoretical_total_throughputs[i]) * 100
        
        data.append((
            bs, 
            round(theoretical_total_throughputs[i], 2),
            None if actual_total is None else round(actual_total, 2),
            None if efficiency is None else round(efficiency, 2),
            bottlenecks[i],
            round(compute_times[i], 6),
            round(memory_times[i], 6)
        ))
    
    print("\\nThroughput and Bottleneck Data:")
    print(f"{'Batch Size':<12} {'Theo. Total':<15} {'Actual Total':<15} {'Efficiency %':<15} {'Bottleneck':<15} {'Compute Time':<15} {'Memory Time':<15}")
    print("-" * 110)
    for bs, tt, at, eff, bot, ct, mt in data:
        at_str = f"{at:.2f}" if at is not None else "N/A"
        eff_str = f"{eff:.2f}%" if eff is not None else "N/A"
        print(f"{bs:<12} {tt:<15.2f} {at_str:<15} {eff_str:<15} {bot:<15} {ct:<15.6f} {mt:<15.6f}")
    
    # Add more space between plots
    plt.tight_layout(pad=4.0)
    
    # Save the figures to files with higher resolution
    fig.savefig(f"throughput_comparison_{model.model.name.replace(' ', '_')}_{model.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png", 
                dpi=300, bbox_inches='tight')
    fig2.savefig(f"bottleneck_analysis_{model.model.name.replace(' ', '_')}_{model.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png", 
                dpi=300, bbox_inches='tight')
    fig3.savefig(f"efficiency_analysis_{model.model.name.replace(' ', '_')}_{model.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png", 
                dpi=300, bbox_inches='tight')
    
    print(f"\\nFigures saved as:")
    print(f"- throughput_comparison_{model.model.name.replace(' ', '_')}_{model.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png")
    print(f"- bottleneck_analysis_{model.model.name.replace(' ', '_')}_{model.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png")
    print(f"- efficiency_analysis_{model.model.name.replace(' ', '_')}_{model.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png")
    
    return fig, fig2, fig3, data