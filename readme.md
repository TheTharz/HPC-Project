# Applying Image Kernels Using CUDA, OpenMP, and Hybrid Approaches

**Author:** Jayawardhana M.V.T.I  
**Index Number:** EG/2020/3996

## 📌 Objective

This project aims to apply commonly used image kernels to digital images using three different parallel computing approaches:

- **CUDA** (GPU-based parallelism)
- **OpenMP** (multi-threaded CPU parallelism)
- **Hybrid** (CUDA + OpenMP for combined CPU-GPU parallelism)

The performance of each approach will be evaluated and compared to a **Serial implementation** to highlight the benefits of parallel computing in image processing tasks.

---

## 🧠 Background

Image kernels (filters) are a fundamental part of many image processing pipelines. Common tasks like:

- **Grayscale conversion**
- **Blurring**
- **Edge detection**

...can be computationally expensive when processing high-resolution images or large datasets. This project explores how **parallel computing** can improve performance:

- **CUDA** uses the GPU's parallel threads to process image pixels concurrently.
- **OpenMP** enables multi-core CPU processing using threads.
- **Hybrid model** combines OpenMP and CUDA for greater efficiency—OpenMP handles image-level parallelism across CPU threads, each invoking CUDA for pixel-level operations.

---

## 🛠️ Methodology

### ✔️ Selected Image Kernels

1. **Grayscale Conversion**

   - Simple per-pixel operation to convert RGB to intensity.

2. **Gaussian Blur**

   - Convolution-based smoothing filter using a floating-point kernel.
   - Helps reduce image noise and detail.

3. **Sobel Edge Detection (X and Y directions)**
   - Detects edges using gradient convolution kernels.

---

### ⚙️ Implementations

Each kernel will be implemented in **four versions**:

1. **Serial Version**

   - Baseline version using standard sequential loops.
   - Processes each pixel one at a time.

2. **CUDA Version**

   - GPU threads process pixels in parallel (1 thread per pixel).
   - Shared memory is used for tiles during Gaussian and Sobel operations.

3. **OpenMP Version**

   - Multi-threaded CPU approach.
   - Parallelizes over image regions and batches using OpenMP.

4. **Hybrid Version**
   - CPU threads (via OpenMP) handle batches of images.
   - Each thread invokes CUDA kernels to process images on the GPU.

---

## 🧪 Performance Evaluation

Execution time of each version will be measured to analyze the speedup gained from parallelism:

- **Metrics:** Execution time, speedup factor compared to serial.
- **Tools:** Time profiling and benchmarking.
- **Platform:** Tested on machines with both multi-core CPUs and CUDA-capable GPUs.

---
