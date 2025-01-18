# Accelerated Exotic Option Pricing


This project demonstrates two models for accelerating exotic option pricing using NVIDIA GPUs.
	1.	Monte Carlo Method (monte-carlo-method.cu): Implemented in CUDA C/C++, this model achieves optimal performance for pricing options by simulating millions of paths on GPUs but requires explicit memory management and boilerplate code. Python GPU libraries like CuPy and RAPIDS streamline this process, enabling scalable and distributed computations with minimal performance loss.
	2.	Deep Derivative Method (deep-derivative-method.py): Using a deep neural network to approximate option pricing, this method delivers up to 35x speed improvements in inference time while maintaining accuracy. The differentiable neural network facilitates efficient calculation of Greeks, with further optimization achievable via TensorRT for production-grade performance.
