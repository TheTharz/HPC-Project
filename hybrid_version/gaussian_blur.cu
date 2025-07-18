#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <omp.h>

int has_image_extension(const char *filename) {
    const char *ext = strrchr(filename, '.');
    if (!ext) return 0;
    return strcmp(ext, ".jpg") == 0 || strcmp(ext, ".png") == 0;
}

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

int process_image_cuda(const char *input_path, const char *output_path) {
    int width, height, channels;
    uint8_t *h_img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!h_img) {
        printf("Failed to load image: %s\n", input_path);
        return 1;
    }

    size_t img_size = width * height * 3;

    uint8_t *d_input, *d_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);

    cudaMemcpy(d_input, h_img, img_size, cudaMemcpyHostToDevice);

    const int h_kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(h_kernel));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gaussian_blur_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for %s: %.3f ms\n", input_path, milliseconds);

    uint8_t *h_output = (uint8_t *)malloc(img_size);
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    if (!stbi_write_png(output_path, width, height, 3, h_output, width * 3)) {
        printf("Failed to write image: %s\n", output_path);
    } else {
        printf("Processed %s → %s\n", input_path, output_path);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_img);
    free(h_output);

    return 0;
}

int main(int argc, char *argv[]) {
    // const char *input_folder = "../images/testing_images";
    const char *input_folder = getenv("INPUT_DIR");
    if (argc > 1) {
        input_folder = argv[1];
    }
    if (!input_folder) {
        fprintf(stderr, "INPUT_DIR not set and no input folder given\n");
        return 1;
    }
    
    const char *output_folder = getenv("OUTPUT_DIR");
    if (argc > 2) {
        output_folder = argv[2];
    }
    if (!output_folder) {
        fprintf(stderr, "OUTPUT_DIR not set and no output folder given\n");
        return 1;
    }
    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        if (mkdir(output_folder, 0755) != 0) {
            perror("Failed to create output directory");
            return 1;
        }
    }

    DIR *dir = opendir(input_folder);
    if (!dir) {
        perror("Failed to open input directory");
        return 1;
    }

    struct dirent *entry;
    char *filenames[1024];
    int count = 0;

    while ((entry = readdir(dir)) != NULL && count < 1024) {
        if (entry->d_type == DT_REG && has_image_extension(entry->d_name)) {
            filenames[count] = strdup(entry->d_name); // Safe copy
            count++;
        }
    }
    closedir(dir);

    double start_time = omp_get_wtime();

    // Parallelized image processing loop
    #pragma omp parallel for
    for (int i = 0; i < count; i++) {
        char input_path[512], output_path[512];
        snprintf(input_path, sizeof(input_path), "%s/%s", input_folder, filenames[i]);
        snprintf(output_path, sizeof(output_path), "%s/gray_%s", output_folder, filenames[i]);
        process_image_cuda(input_path, output_path);
        free(filenames[i]); // cleanup
    }

    double end_time = omp_get_wtime();
    printf("Total processing time: %.3f seconds with %d threads\n", end_time - start_time, omp_get_max_threads());
    printf("Total processing time: %f seconds\n", end_time - start_time);

    return 0;
}
