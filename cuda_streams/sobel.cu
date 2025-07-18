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
#define NUM_STREAMS 4  

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

uint8_t* sobel_gpu_async(uint8_t* input, int width, int height, cudaStream_t stream) {
    size_t img_size = width * height * sizeof(uint8_t);
    uint8_t *d_input = NULL, *d_output = NULL;
    uint8_t *output = (uint8_t*)malloc(img_size);
    if (!output) return NULL;

    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_output, img_size);

    cudaMemcpyAsync(d_input, input, img_size, cudaMemcpyHostToDevice, stream);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    sobel_kernel<<<gridDim, blockDim, 0, stream>>>(d_input, d_output, width, height);

    cudaMemcpyAsync(output, d_output, img_size, cudaMemcpyDeviceToHost, stream);

    // Free device memory after stream finishes
    cudaStreamAddCallback(stream,
        [](cudaStream_t stream, cudaError_t status, void* userData) {
            uint8_t **devPtrs = (uint8_t**)userData;
            cudaFree(devPtrs[0]);
            cudaFree(devPtrs[1]);
            free(devPtrs);
        }, 
        malloc(sizeof(uint8_t*) * 2), 0);

    return output;
}

int has_image_extension(const char *filename) {
    const char *ext = strrchr(filename, '.');
    return ext && (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".png") == 0);
}

typedef struct {
    char output_path[512];
    int width;
    int height;
    uint8_t *output_buffer;
} ImageResult;

int main(int argc, char *argv[]) {
    double total_start_time = omp_get_wtime();

    const char *input_folder = getenv("INPUT_DIR");
    if (argc > 1) input_folder = argv[1];
    if (!input_folder) {
        fprintf(stderr, "INPUT_DIR not set and no input folder given\n");
        return 1;
    }

    const char *output_folder = getenv("OUTPUT_DIR");
    if (argc > 2) output_folder = argv[2];
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

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    struct dirent *entries[1024];
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL && count < 1024) {
        if (entry->d_type == DT_REG && has_image_extension(entry->d_name)) {
            entries[count] = (struct dirent*)malloc(sizeof(struct dirent));
            memcpy(entries[count], entry, sizeof(struct dirent));
            count++;
        }
    }
    closedir(dir);

    ImageResult results[1024] = {0};

    for (int i = 0; i < count; i++) {
        int stream_id = i % NUM_STREAMS;

        char input_path[512], output_path[512];
        snprintf(input_path, sizeof(input_path), "%s/%s", input_folder, entries[i]->d_name);
        snprintf(output_path, sizeof(output_path), "%s/sobel_%s", output_folder, entries[i]->d_name);

        int width, height, channels;
        unsigned char *img = stbi_load(input_path, &width, &height, &channels, 1);
        if (!img) {
            fprintf(stderr, "Failed to load %s\n", input_path);
            free(entries[i]);
            continue;
        }

        uint8_t *sobel = sobel_gpu_async(img, width, height, streams[stream_id]);
        stbi_image_free(img);

        strncpy(results[i].output_path, output_path, sizeof(results[i].output_path) - 1);
        results[i].width = width;
        results[i].height = height;
        results[i].output_buffer = sobel;

        free(entries[i]);
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Write images to disk
    for (int i = 0; i < count; i++) {
        if (results[i].output_buffer) {
            int stride_in_bytes = results[i].width * 1;
            if (!stbi_write_png(results[i].output_path, results[i].width, results[i].height, 1, results[i].output_buffer, stride_in_bytes)) {
                fprintf(stderr, "Failed to save %s\n", results[i].output_path);
            }
            free(results[i].output_buffer);
        }
    }

    // Destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    double total_end_time = omp_get_wtime();
    printf("Total time: %f seconds\n", total_end_time - total_start_time);

    return 0;
}
