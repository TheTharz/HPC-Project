#!/bin/bash

# === Compile Serial Versions ===
cd serial_version

for file in gray_scale_filter gaussian_blur sobel laplacian_filter; do
  dst=../../web-platform/binaries/serial/$file
  [ -f "$dst" ] && rm "$dst"
  gcc -fopenmp -o "$dst" "$file.c" -lm
done

# === Compile OpenMP Versions ===
cd ../openmp_version

for file in gray_scale_filter gaussian_blur sobel laplacian_filter; do
  dst=../../web-platform/binaries/openmp/$file
  [ -f "$dst" ] && rm "$dst"
  gcc -fopenmp -o "$dst" "$file.c" -lm
done

# === Compile CUDA Versions ===
cd ../cuda_version

for file in gray_scale_filter gaussian_blur sobel laplacian_filter; do
  dst=../../web-platform/binaries/cuda/$file
  [ -f "$dst" ] && rm "$dst"
  nvcc -Xcompiler -fopenmp -o "$dst" "$file.cu"
done

# === Compile Hybrid Versions ===
cd ../hybrid_version

for file in gray_scale_filter gaussian_blur sobel laplacian_filter; do
  dst=../../web-platform/binaries/hybrid/$file
  [ -f "$dst" ] && rm "$dst"
  nvcc -Xcompiler -fopenmp -o "$dst" "$file.cu"
done
