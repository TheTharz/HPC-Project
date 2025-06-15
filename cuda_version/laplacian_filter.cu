#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

// 3x3 Laplacian kernel in constant memory
__constant__ int d_kernel[3][3];

// CUDA kernel to apply Laplacian filter
__global__ void laplacian_kernel(uint8_t *input, uint8_t *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int sum = 0;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int nx = x + kx;
            int ny = y + ky;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int pixel = input[ny * width + nx];
                int weight = d_kernel[ky + 1][kx + 1];
                sum += pixel * weight;
            }
        }
    }

    // Clamp result to [0, 255]
    sum = min(max(sum, 0), 255);
    output[y * width + x] = (uint8_t)sum;
}

int main() {
    int width, height, channels;
    uint8_t *h_img = stbi_load("../images/test.jpg", &width, &height, &channels, 1); // force grayscale
    if (!h_img) {
        printf("Failed to load image\n");
        return 1;
    }

    size_t img_size = width * height * sizeof(uint8_t);

    // Allocate device memory
    uint8_t *d_input, *d_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);

    // Copy image to device
    cudaMemcpy(d_input, h_img, img_size, cudaMemcpyHostToDevice);

    // Copy Laplacian kernel to device constant memory
    int h_kernel[3][3] = {
        {  0, -1,  0 },
        { -1,  4, -1 },
        {  0, -1,  0 }
    };
    cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(h_kernel));

    // Launch CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    laplacian_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // Copy result back to host
    uint8_t *h_output = (uint8_t *)malloc(img_size);
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    // Save image
    if (!stbi_write_png("output/laplacian_cuda.png", width, height, 1, h_output, width)) {
        printf("Failed to write image\n");
    } else {
        printf("Laplacian image saved to output/laplacian_cuda.png\n");
    }

    printf("Processed %d pixels in %.3f ms using CUDA\n", width * height, elapsed_ms);

    // Cleanup
    stbi_image_free(h_img);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
