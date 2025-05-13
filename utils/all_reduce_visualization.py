import torch

BANDWIDTH_H100 = 18 * 26.5 * (1024*1024*1024) # 18 links, 26.5GB/s per link
BANDWIDTH_4090 = 32 * (1024*1024*1024) # 18 links, 26.5GB/s per link

batch_size, seqlen, hiden_dim = 16, 2048, 1024
bytes_per_datatype = 2

data = batch_size * seqlen * hiden_dim * bytes_per_datatype 

latency_ms = data/BANDWIDTH_4090 * 1000

print(f"Latency for {data / (1024 * 1024 * 1024)} GB of data: {latency_ms:.2f} ms")

