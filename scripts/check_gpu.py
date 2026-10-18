"""Quick CUDA check for classroom-noise-ai (run inside the venv):  python scripts/check_gpu.py"""
from __future__ import annotations

import time

try:
    import torch
except ImportError:
    raise SystemExit("torch is not installed in this environment")

print(f"torch {torch.__version__} | built for CUDA: {torch.version.cuda}")
if not torch.cuda.is_available():
    print("cuda_available = False")
    if torch.version.cuda is None:
        print("-> this is a CPU-ONLY build of torch (version ends with '+cpu' or has no '+cuXXX'). "
              "Reinstall from the CUDA index (see README, section 'GPU on Windows').")
    else:
        print("-> CUDA build, but no usable GPU/driver: update the NVIDIA driver and run nvidia-smi.")
    raise SystemExit(1)

props = torch.cuda.get_device_properties(0)
print(f"cuda_available = True | {props.name} | {props.total_memory / 2**30:.1f} GB | "
      f"compute capability {props.major}.{props.minor}")
x = torch.randn(2048, 2048, device="cuda")
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(20):
    x @ x
torch.cuda.synchronize()
print(f"matmul test OK ({(time.perf_counter() - t) * 1000 / 20:.2f} ms per 2048x2048 product)")
