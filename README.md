# LLM_Workload_Characterization
This project explores the computational performance characteristics of open-source Large
Language Models (LLMs) through systematic empirical analysis. By leveraging an RTX 5090
GPU and industry-standard profiling instrumentation, we will generate novel insights into the
execution dynamics, resource utilization patterns, and optimization opportunities across model
architectures and inference frameworks.

## Objectives  
**Primary Focus:** Quantify compute performance across dense and mixture-of-experts (MoE)
models and benchmark inference performance across major frameworks (Llama.cpp, Ollama,
vLLM, SGLang). Analyse compute performance through multi-layer profiling, revealing the
critical execution path(s), GPU occupancy dynamics, memory bandwidth utilization, and cache
efficiency characteristics. Characterize the CPU/GPU compute flow and identify memory-
compute bottlenecks. Identify optimizations to improve inference compute performance.
