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
#include <omp.h>

__global__ void grayscale_kernel(uint8_t *input, uint8_t *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        uint8_t r = input[idx];
        uint8_t g = input[idx + 1];
        uint8_t b = input[idx + 2];
        output[y * width + x] = (uint8_t)(0.3f * r + 0.59f * g + 0.11f * b);
    }
}

int has_image_extension(const char *filename) {
    const char *ext = strrchr(filename, '.');
    if (!ext) return 0;
    return strcmp(ext, ".jpg") == 0 || strcmp(ext, ".png") == 0;
}

void process_image_cuda(const char *input_path, const char *output_path) {
    int width, height, channels;
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!img) {
        #pragma omp critical
        printf("Failed to load image: %s\n", input_path);
        return;
    }

    size_t img_size_rgb = width * height * 3;
    size_t img_size_gray = width * height;

    uint8_t *d_input, *d_output;
    cudaMalloc((void **)&d_input, img_size_rgb);
    cudaMalloc((void **)&d_output, img_size_gray);
    cudaMemcpy(d_input, img, img_size_rgb, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + 15) / 16, (height + 15) / 16);
    grayscale_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    uint8_t *gray_img = (uint8_t *)malloc(img_size_gray);
    cudaMemcpy(gray_img, d_output, img_size_gray, cudaMemcpyDeviceToHost);

    if (!stbi_write_png(output_path, width, height, 1, gray_img, width)) {
        #pragma omp critical
        printf("Failed to write image: %s\n", output_path);
    } else {
        #pragma omp critical
        printf("Processed %s → %s\n", input_path, output_path);
    }

    stbi_image_free(img);
    free(gray_img);
    cudaFree(d_input);
    cudaFree(d_output);
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

    // Collect image filenames
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
