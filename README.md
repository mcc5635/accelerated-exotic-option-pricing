# Accelerated Exotic Option Pricing

This project demonstrates two models for accelerating exotic option pricing using NVIDIA GPUs:

### 1. Monte Carlo Method (`monte-carlo-method.cu`)
- **Implementation**: CUDA C/C++
- **Overview**: Achieves optimal performance for pricing options by simulating millions of paths on GPUs.
- **Details**:
  - Requires explicit memory management and boilerplate code.
  - Python GPU libraries like CuPy and RAPIDS simplify the process, enabling scalable and distributed computations with minimal performance loss.

---

### 2. Deep Derivative Method (`deep-derivative-method.py`)
- **Implementation**: Python with PyTorch
- **Overview**: Uses a deep neural network to approximate option pricing for significant speed improvements.
- **Details**:
  - Delivers up to **35x speed improvements** in inference time while maintaining accuracy.
  - Facilitates efficient calculation of Greeks using a fully differentiable neural network.
  - Further optimization achievable via **TensorRT** for production-grade performance.

---

These models highlight the power of GPU acceleration in quantitative finance, offering a blend of computational efficiency and simplicity for research and production environments.
