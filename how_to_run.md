add a image to the images folder and create a output folder
gcc grey_scale_filter.c -o grey_scale_filter -lm

nvcc -Xcompiler -fopenmp laplacian_filter.cu -o laplacian_filter.o -lm
gcc -fopenmp sobel.c -o sobel -lm

nvcc -Xcompiler -fopenmp -o sobel sobel.cu -O2

# Create binary directories

mkdir -p server/binaries/{serial,openmp,cuda,hybrid}

# Compile serial versions

cd HPC-Project/serial_version
gcc -o ../../web-platform/server/binaries/serial/gray_scale_filter gray_scale_filter.c -lm
gcc -o ../../web-platform/server/binaries/serial/gaussian_blur gaussian_blur.c -lm
gcc -o ../../web-platform/server/binaries/serial/sobel sobel.c -lm
gcc -o ../../web-platform/server/binaries/serial/laplacian_filter laplacian_filter.c -lm

# Compile OpenMP versions

cd ../openmp_version
gcc -fopenmp -o ../../web-platform/server/binaries/openmp/gray_scale_filter gray_scale_filter.c -lm
gcc -fopenmp -o ../../web-platform/server/binaries/openmp/gaussian_blur gaussian_blur.c -lm
gcc -fopenmp -o ../../web-platform/server/binaries/openmp/sobel sobel.c -lm
gcc -fopenmp -o ../../web-platform/server/binaries/openmp/laplacian_filter laplacian_filter.c -lm

# Compile CUDA versions

cd ../cuda_version
nvcc -o ../../web-platform/server/binaries/cuda/gray_scale_filter gray_scale_filter.cu
nvcc -o ../../web-platform/server/binaries/cuda/gaussian_blur gaussian_blur.cu
nvcc -o ../../web-platform/server/binaries/cuda/sobel sobel.cu
nvcc -o ../../web-platform/server/binaries/cuda/laplacian_filter laplacian_filter.cu

# Compile Hybrid versions

cd ../hybrid_version
nvcc -Xcompiler -fopenmp -o ../../web-platform/server/binaries/hybrid/gray_scale_filter gray_scale_filter.cu
nvcc -Xcompiler -fopenmp -o ../../web-platform/server/binaries/hybrid/gaussian_blur gaussian_blur.cu
nvcc -Xcompiler -fopenmp -o ../../web-platform/server/binaries/hybrid/sobel sobel.cu
nvcc -Xcompiler -fopenmp -o ../../web-platform/server/binaries/hybrid/laplacian_filter laplacian_filter.cu
