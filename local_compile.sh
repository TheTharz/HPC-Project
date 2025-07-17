#!/bin/bash

# === Compile Serial Versions ===
cd serial_version

for file in gray_scale_filter gaussian_blur sobel laplacian_filter; do
  dst=../binaries/serial/$file
  if [ -f "$dst" ]; then
    rm "$dst"
  fi
  gcc -fopenmp -o "$dst" "$file.c" -lm
done

# === Compile OpenMP Versions ===
cd ../openmp_version

for file in gray_scale_filter gaussian_blur sobel laplacian_filter; do
  dst=../binaries/openmp/$file
  if [ -f "$dst" ]; then
    rm "$dst"
  fi
  gcc -fopenmp -o "$dst" "$file.c" -lm
done

# === Compile CUDA Versions ===
cd ../cuda_version

for file in gray_scale_filter gaussian_blur sobel laplacian_filter; do
  dst=../binaries/cuda/$file
  if [ -f "$dst" ]; then
    rm "$dst"
  fi
  nvcc -Xcompiler -fopenmp -o "$dst" "$file.cu"
done

# === Compile Hybrid Versions ===
cd ../hybrid_version

for file in gray_scale_filter gaussian_blur sobel laplacian_filter; do
  dst=../binaries/hybrid/$file
  if [ -f "$dst" ]; then
    rm "$dst"
  fi
  nvcc -Xcompiler -fopenmp -o "$dst" "$file.cu"
done

