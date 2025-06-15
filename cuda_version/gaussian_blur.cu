#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__constant__ int d_kernel[3][3];  // kernel in constant memory

__global__ void gaussian_blur_kernel(uint8_t *input, uint8_t *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int sum_r = 0, sum_g = 0, sum_b = 0;
    int weight_sum = 0;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int nx = x + kx;
            int ny = y + ky;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int idx = (ny * width + nx) * 3;
                int weight = d_kernel[ky + 1][kx + 1];

                sum_r += input[idx] * weight;
                sum_g += input[idx + 1] * weight;
                sum_b += input[idx + 2] * weight;
                weight_sum += weight;
            }
        }
    }

    int out_idx = (y * width + x) * 3;
    output[out_idx]     = min(max(sum_r / weight_sum, 0), 255);
    output[out_idx + 1] = min(max(sum_g / weight_sum, 0), 255);
    output[out_idx + 2] = min(max(sum_b / weight_sum, 0), 255);
}

int main() {
    // Load image
    int width, height, channels;
    uint8_t *h_img = stbi_load("../images/test.jpg", &width, &height, &channels, 3);
    if (!h_img) {
        printf("Failed to load image\n");
        return 1;
    }

    size_t img_size = width * height * 3;

    // Allocate device memory
    uint8_t *d_input, *d_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);

    // Copy image to device
    cudaMemcpy(d_input, h_img, img_size, cudaMemcpyHostToDevice);

    // Copy kernel to constant memory
    const int h_kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(h_kernel));

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gaussian_blur_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    uint8_t *h_output = (uint8_t *)malloc(img_size);
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    // Save output image
    if (!stbi_write_png("output/gaussian_cuda.png", width, height, 3, h_output, width * 3)) {
        printf("Failed to write image\n");
    } else {
        printf("CUDA Gaussian image saved to gaussian_cuda.png\n");
    }

    printf("Processed %d pixels in %f ms using CUDA\n", width * height, milliseconds);

    // Free resources
    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_img);
    free(h_output);

    return 0;
}
