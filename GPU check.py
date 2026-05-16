import sys

import torch
print(f"GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}, {p.total_memory/1e9:.1f} GB")

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_capability(0))


import torch
# Simulate memory usage at batch_size=48, 120x160 fp16
b, c, h, w = 48, 3, 128, 176  # padded size
x = torch.randn(b, c, h, w, dtype=torch.float16, device='cuda:0')
print(f'Input tensor: {x.element_size() * x.nelement() / 1e6:.1f} MB')
print(f'Free VRAM: {torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB')
print(sys.version)