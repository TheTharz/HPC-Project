#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Define Sobel kernels in constant memory
__constant__ int d_Gx[3][3];
__constant__ int d_Gy[3][3];

// CUDA kernel for Sobel edge detection (grayscale image)
__global__ void sobel_kernel(uint8_t *input, uint8_t *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int Gx = 0, Gy = 0;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int nx = x + kx;
            int ny = y + ky;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int pixel = input[ny * width + nx];
                Gx += pixel * d_Gx[ky + 1][kx + 1];
                Gy += pixel * d_Gy[ky + 1][kx + 1];
            }
        }
    }

    int magnitude = (int)sqrtf((float)(Gx * Gx + Gy * Gy));
    magnitude = min(max(magnitude, 0), 255);
    output[y * width + x] = (uint8_t)magnitude;
}

int main() {
    int width, height, channels;
    uint8_t *h_img = stbi_load("../images/gray_image.png", &width, &height, &channels, 1); // force grayscale
    if (!h_img) {
        printf("Failed to load image\n");
        return 1;
    }

    size_t img_size = width * height;

    // Allocate device memory
    uint8_t *d_input, *d_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);

    // Copy image to device
    cudaMemcpy(d_input, h_img, img_size, cudaMemcpyHostToDevice);

    // Define Sobel kernels on host
    int h_Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int h_Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    cudaMemcpyToSymbol(d_Gx, h_Gx, sizeof(h_Gx));
    cudaMemcpyToSymbol(d_Gy, h_Gy, sizeof(h_Gy));

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sobel_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    uint8_t *h_output = (uint8_t *)malloc(img_size);
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    // Save result
    if (!stbi_write_png("output/sobel_cuda.png", width, height, 1, h_output, width)) {
        printf("Failed to write image\n");
    } else {
        printf("Sobel image saved to output/sobel_cuda.png\n");
    }

    printf("Processed %d pixels in %.3f ms using CUDA\n", width * height, milliseconds);

    // Cleanup
    stbi_image_free(h_img);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
