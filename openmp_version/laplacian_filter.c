#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>

#include<omp.h>

int has_image_extension(const char *filename) {
    const char *ext = strrchr(filename, '.');
    if (!ext) return 0;
    return strcmp(ext, ".jpg") == 0 || strcmp(ext, ".png") == 0;
}

uint8_t* apply_laplacian_filter(uint8_t *gray_img, int width, int height) {
    int kernel[3][3] = {
        {  0, -1,  0 },
        { -1,  4, -1 },
        {  0, -1,  0 }
    };

    uint8_t *output = malloc(width * height);
    if (!output) return NULL;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int pixel = gray_img[ny * width + nx];
                        sum += pixel * kernel[ky + 1][kx + 1];
                    }
                }
            }

            // Clamp to [0, 255]
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;

            output[y * width + x] = (uint8_t)sum;
        }
    }

    return output;
}

int main(int argc, char *argv[]) {
    int num_threads = 8; 
    omp_set_num_threads(num_threads);
    printf("Using %d threads\n", num_threads);
    
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

    struct dirent *entries[1024];
    int entry_count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL && entry_count < 1024) {
        if (entry->d_type == DT_REG && has_image_extension(entry->d_name)) {
            entries[entry_count] = malloc(sizeof(struct dirent));
            if (entries[entry_count]) {
                memcpy(entries[entry_count], entry, sizeof(struct dirent));
                entry_count++;
            }
        }
    }
    closedir(dir);

    double total_start_time = omp_get_wtime(); // Start total timer

    #pragma omp parallel for
    for (int i = 0; i < entry_count; i++) {
        char input_path[512];
        snprintf(input_path, sizeof(input_path), "%s/%s", input_folder, entries[i]->d_name);

        int width, height, channels;
        unsigned char *img = stbi_load(input_path, &width, &height, &channels, 3);
        if (!img) {
            printf("Failed to load image: %s\n", input_path);
            free(entries[i]);
            continue;
        }

        double start_time = omp_get_wtime();
        uint8_t *gray_img = apply_laplacian_filter(img, width, height);
        double end_time = omp_get_wtime();

        if (!gray_img) {
            printf("Memory allocation failed for %s\n", entries[i]->d_name);
            stbi_image_free(img);
            free(entries[i]);
            continue;
        }

        char output_path[512];
        snprintf(output_path, sizeof(output_path), "%s/laplacian_%s",output_folder, entries[i]->d_name);

        if (!stbi_write_png(output_path, width, height, 1, gray_img, width)) {
            printf("Failed to write image: %s\n", output_path);
        } else {
            printf("Processed %s in %f seconds\n", entries[i]->d_name, end_time - start_time);
        }

        stbi_image_free(img);
        free(gray_img);
        free(entries[i]);
    }

    double total_end_time = omp_get_wtime(); // End total timer
    printf("Total processing time: %f seconds\n", total_end_time - total_start_time);

    return 0;
}
