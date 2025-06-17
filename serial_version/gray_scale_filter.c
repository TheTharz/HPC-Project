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

uint8_t* convert_to_grayscale(unsigned char *img, int width, int height) {
    uint8_t *gray_img = malloc(width * height);
    if (!gray_img) {
        return NULL;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            uint8_t r = img[idx];
            uint8_t g = img[idx + 1];
            uint8_t b = img[idx + 2];
            gray_img[y * width + x] = (uint8_t)(0.3 * r + 0.59 * g + 0.11 * b);
        }
    }

    return gray_img;
}

int main() {
    const char *input_folder = "../images/testing_images";

    // Create output directory if it doesn't exist
    struct stat st = {0};
    if (stat("output", &st) == -1) {
        if (mkdir("output", 0755) != 0) {
            perror("Failed to create output directory");
            return 1;
        }
    }

    DIR *dir = opendir(input_folder);
    if (!dir) {
        perror("Failed to open input directory");
        return 1;
    }

    double total_start_time = omp_get_wtime(); // Start total timer

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG && has_image_extension(entry->d_name)) {
            char input_path[512];
            snprintf(input_path, sizeof(input_path), "%s/%s", input_folder, entry->d_name);

            int width, height, channels;
            unsigned char *img = stbi_load(input_path, &width, &height, &channels, 3);
            if (!img) {
                printf("Failed to load image: %s\n", input_path);
                continue;
            }

            double start_time = omp_get_wtime();
            uint8_t *gray_img = convert_to_grayscale(img, width, height);
            double end_time = omp_get_wtime();

            if (!gray_img) {
                printf("Memory allocation failed for %s\n", entry->d_name);
                stbi_image_free(img);
                continue;
            }

            char output_path[512];
            snprintf(output_path, sizeof(output_path), "output/gray_%s", entry->d_name);

            if (!stbi_write_png(output_path, width, height, 1, gray_img, width)) {
                printf("Failed to write image: %s\n", output_path);
            } else {
                printf("Processed %s in %f seconds\n", entry->d_name, end_time - start_time);
            }

            stbi_image_free(img);
            free(gray_img);
        }
    }
    closedir(dir);

    double total_end_time = omp_get_wtime(); // End total timer
    printf("Total processing time: %f seconds\n", total_end_time - total_start_time);

    return 0;
}
