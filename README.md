# ML-with-GPU-Acceleration

## Key Learnings:
**1. What is CUDA?**
- CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model.
- It acts as the software layer and instruction set that allows general-purpose programming languages (like C++ and Python) to directly utilize the thousands of small, parallel processing cores on an NVIDIA GPU.
- **Impact**: CUDA is the indispensable foundation for achieving the massive speedups seen in modern AI/ML workloads.

**2. cuDF vs Pandas**
- GPU Usage: cuDF is specifically built to run on the GPU. Pandas runs exclusively on the CPU and cannot be accelerated by the GPU.
- Takeaway: By replacing Pandas operations with cuDF, we kept the data on the GPU for the entire data preparation phase, minimizing slow data transfers.

**3. What is DMatrix and how does it fit in with GPU acceleration and XGBoost?**
- DMatrix (Data Matrix) is the internal, optimized data structure used by the XGBoost library.
- It's a highly efficient format that enables XGBoost to quickly access and process data during the tree-building process.
- **Acceleration**: When paired with the tree_method='gpu_hist' parameter, the DMatrix uses GPU memory, allowing the core training algorithm to run entirely on the parallel CUDA cores.

**4. GPU-CPU Data Transfer (The .get() Method)**
- The .get() method (used on cuDF or CuPy objects) forces the data to be copied from the fast GPU VRAM back to the slower CPU RAM.
- In this project: We used .get() only when setting up the CPU baseline model (xgb.DMatrix(X_train.get(), ...)), as the standard CPU XGBoost requires data in CPU memory.
- **Key Insight**: This transfer is the biggest bottleneck (known as the PCIe Bus Transfer). The goal of acceleration is to minimize or eliminate these transfers by running the entire pipeline on the GPU.

## Performance & Scaling Takeaways
**5. Why Does Speedup Scale with Data Size?**
- GPU acceleration provides the greatest benefit when dealing with large volumes of data because it maximizes the utility of the GPU's parallelism.
- With Small Data: The time needed to launch the parallel tasks on the GPU can offset the computational gain.
- With Large Data (As Demonstrated): The GPU's ability to process thousands / millions of rows simultaneously in parallel dramatically reduces the total runtime. **The experiment demonstrated** that the **GPU's advantage becomes significantly more pronounced** as the data size increases, proving the value of acceleration for real-world scenarios.
