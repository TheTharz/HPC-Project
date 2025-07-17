#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <cuda_runtime.h>
#include <omp.h>

#define BLOCK_SIZE 16

__global__ void sobel_kernel(const uint8_t* input, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    if (x >= width || y >= height) return;

    int sumX = 0, sumY = 0;
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int nx = x + kx;
            int ny = y + ky;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int idx = ny * width + nx;
                int pixel = input[idx];
                sumX += pixel * Gx[ky + 1][kx + 1];
                sumY += pixel * Gy[ky + 1][kx + 1];
            }
        }
    }

    int magnitude = sqrtf((float)(sumX * sumX + sumY * sumY));
    if (magnitude > 255) magnitude = 255;
    if (magnitude < 0) magnitude = 0;

    output[y * width + x] = (uint8_t)magnitude;
}

uint8_t* sobel_gpu(uint8_t* input, int width, int height) {
    size_t img_size = width * height * sizeof(uint8_t);
    uint8_t *d_input, *d_output, *output = (uint8_t*)malloc(img_size);

    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_output, img_size);
    cudaMemcpy(d_input, input, img_size, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    sobel_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, img_size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

int has_image_extension(const char *filename) {
    const char *ext = strrchr(filename, '.');
    return ext && (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".png") == 0);
}

int main(int argc, char *argv[]) {
    double total_start_time = omp_get_wtime();

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

    // Step 1: Load all image file names into a list
    struct dirent *entries[1024];
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL && count < 1024) {
        if (entry->d_type == DT_REG && has_image_extension(entry->d_name)) {
            entries[count] = (struct dirent*) malloc(sizeof(struct dirent));
            memcpy(entries[count], entry, sizeof(struct dirent));
            count++;
        }
    }
    closedir(dir);

    // Step 2: Parallelize image processing
    #pragma omp parallel for
    for (int i = 0; i < count; i++) {
        char input_path[512], output_path[512];
        snprintf(input_path, sizeof(input_path), "%s/%s", input_folder, entries[i]->d_name);

        int width, height, channels;
        unsigned char *img = stbi_load(input_path, &width, &height, &channels, 1);
        if (!img) {
            #pragma omp critical
            fprintf(stderr, "Failed to load %s\n", input_path);
            free(entries[i]);
            continue;
        }

        double start = omp_get_wtime();
        uint8_t *sobel = sobel_gpu(img, width, height);
        double end = omp_get_wtime();

        snprintf(output_path, sizeof(output_path), "%s/sobel_%s",output_folder, entries[i]->d_name);
        if (!stbi_write_png(output_path, width, height, 1, sobel, width)) {
            #pragma omp critical
            fprintf(stderr, "Failed to write %s\n", output_path);
        } else {
            #pragma omp critical
            printf("Thread %d: Processed %s in %f seconds (CUDA)\n", omp_get_thread_num(), entries[i]->d_name, end - start);
        }

        stbi_image_free(img);
        free(sobel);
        free(entries[i]);
    }

    double total_end_time = omp_get_wtime();
    printf("Total processing time: %f seconds\n", total_end_time - total_start_time);
    return 0;
}
