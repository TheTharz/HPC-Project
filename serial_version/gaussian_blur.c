#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <omp.h>
// 3x3 Gaussian kernel with sigma ~1
// Normalized so sum = 16
const int kernel[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

uint8_t* gaussian_blur_rgb(uint8_t *input, int width, int height) {
    uint8_t *output = malloc(width * height * 3);
    if (!output) return NULL;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum_r = 0;
            int sum_g = 0;
            int sum_b = 0;
            int weight_sum = 0;

            // Apply 3x3 kernel with zero-padding
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;

                    // Check for valid neighbors inside image bounds
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int idx = (ny * width + nx) * 3;
                        int r = input[idx];
                        int g = input[idx + 1];
                        int b = input[idx + 2];
                        int weight = kernel[ky + 1][kx + 1];
                        sum_r += r * weight;
                        sum_g += g * weight;
                        sum_b += b * weight;
                        weight_sum += weight;
                    }
                }
            }

            // Normalize and clamp the result
            int val_r = sum_r / weight_sum;
            int val_g = sum_g / weight_sum;
            int val_b = sum_b / weight_sum;
            if (val_r > 255) val_r = 255;
            if (val_g > 255) val_g = 255;
            if (val_b > 255) val_b = 255;
            if (val_r < 0) val_r = 0;
            if (val_g < 0) val_g = 0;
            if (val_b < 0) val_b = 0;

            int idx = (y * width + x) * 3;
            output[idx] = (uint8_t)val_r;
            output[idx + 1] = (uint8_t)val_g;
            output[idx + 2] = (uint8_t)val_b;
        }
    }

    return output;
}

int main() {
    int width, height, channels;
    unsigned char *img = stbi_load("../images/test.jpg", &width, &height, &channels, 0);
    if (!img) {
        printf("Failed to load image\n");
        return 1;
    }

    double start_time = omp_get_wtime();

    uint8_t *rgb_img = gaussian_blur_rgb(img, width, height);
    double end_time = omp_get_wtime();
    if (!rgb_img) {
        printf("Memory allocation failed\n");
        stbi_image_free(img);
        return 1;
    }

    double time_taken= end_time - start_time;
    printf("Processed %d pixels in %f seconds\n", width * height, time_taken);

    if (!stbi_write_png("output/gaussian.png", width, height, 3, rgb_img, width * 3)) {
        printf("Failed to write image\n");
    } else {
        printf("RGB image saved to gaussian_rgb.png\n");
    }

    stbi_image_free(img);
    free(rgb_img);

    return 0;
}

