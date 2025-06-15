#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(uint8_t *rgb, uint8_t *gray, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        int rgb_idx = idx * 3;
        uint8_t r = rgb[rgb_idx];
        uint8_t g = rgb[rgb_idx + 1];
        uint8_t b = rgb[rgb_idx + 2];
        gray[idx] = (uint8_t)(0.3f * r + 0.59f * g + 0.11f * b);
    }
}

int main() {
    int width, height, channels;
    unsigned char *h_rgb = stbi_load("../images/test.jpg", &width, &height, &channels, 3);
    if (!h_rgb) {
        printf("Failed to load image\n");
        return 1;
    }

    int img_size = width * height;
    int rgb_size = img_size * 3;

    uint8_t *d_rgb, *d_gray;
    uint8_t *h_gray = (uint8_t *)malloc(img_size);

    // Allocate device memory
    cudaMalloc((void **)&d_rgb, rgb_size);
    cudaMalloc((void **)&d_gray, img_size);

    // Copy image data to device
    cudaMemcpy(d_rgb, h_rgb, rgb_size, cudaMemcpyHostToDevice);

    // Timing the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (img_size + threadsPerBlock - 1) / threadsPerBlock;
    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_rgb, d_gray, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(h_gray, d_gray, img_size, cudaMemcpyDeviceToHost);

    // Save the image
    if (!stbi_write_png("output/gray_image_cuda.png", width, height, 1, h_gray, width)) {
        printf("Failed to write image\n");
    } else {
        printf("Grayscale image saved to gray_image_cuda.png\n");
    }

    printf("CUDA processing time: %.8f s\n", milliseconds / 1000.0f);

    // Cleanup
    stbi_image_free(h_rgb);
    free(h_gray);
    cudaFree(d_rgb);
    cudaFree(d_gray);

    return 0;
}

