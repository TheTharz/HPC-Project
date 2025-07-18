#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <omp.h>

__constant__ int d_kernel[3][3];  

int has_image_extension(const char *filename) {
    const char *ext = strrchr(filename, '.');
    if (!ext) return 0;
    return strcmp(ext, ".jpg") == 0 || strcmp(ext, ".png") == 0;
}

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

    sum = min(max(sum, 0), 255);
    output[y * width + x] = (uint8_t)sum;
}

int process_image(const char *input_path, const char *output_path) {
    int width, height, channels;
    uint8_t *h_img = stbi_load(input_path, &width, &height, &channels, 1);  // Grayscale
    if (!h_img) {
        printf("Failed to load: %s\n", input_path);
        return 1;
    }

    size_t img_size = width * height;

    uint8_t *d_input, *d_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);

    cudaMemcpy(d_input, h_img, img_size, cudaMemcpyHostToDevice);

    int h_kernel[3][3] = {
        {  0, -1,  0 },
        { -1,  4, -1 },
        {  0, -1,  0 }
    };
    cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(h_kernel));

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

    uint8_t *h_output = (uint8_t *)malloc(img_size);
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    if (!stbi_write_png(output_path, width, height, 1, h_output, width)) {
        printf("Failed to save: %s\n", output_path);
    } else {
        printf("Saved: %s (%.2f ms)\n", output_path, elapsed_ms);
    }

    // Cleanup
    stbi_image_free(h_img);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

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
        perror("Cannot open input folder");
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
        snprintf(output_path, sizeof(output_path), "%s/laplacian_%s", output_folder, filenames[i]);
        process_image(input_path, output_path);
        free(filenames[i]); // cleanup
    }

    double end_time = omp_get_wtime();
    printf("Total processing time: %f seconds\n", end_time - start_time);


    return 0;
}
