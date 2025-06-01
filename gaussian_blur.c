#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// 3x3 Gaussian kernel with sigma ~1
// Normalized so sum = 16
const int kernel[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

uint8_t* gaussian_blur(uint8_t *input, int width, int height) {
    uint8_t *output = malloc(width * height);
    if (!output) return NULL;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum = 0;
            int weight_sum = 0;

            // Apply 3x3 kernel with zero-padding
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;

                    // Check for valid neighbors inside image bounds
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int pixel = input[ny * width + nx];
                        int weight = kernel[ky + 1][kx + 1];
                        sum += pixel * weight;
                        weight_sum += weight;
                    }
                }
            }

            // Normalize and clamp the result
            int val = sum / weight_sum;
            if (val > 255) val = 255;
            if (val < 0) val = 0;

            output[y * width + x] = (uint8_t)val;
        }
    }

    return output;
}

int main() {
    int width, height, channels;
    unsigned char *img = stbi_load("output/gray_image.png", &width, &height, &channels, 3);
    if (!img) {
        printf("Failed to load image\n");
        return 1;
    }

    clock_t start_time = clock();

    uint8_t *gray_img = gaussian_blur(img, width, height);
    if (!gray_img) {
        printf("Memory allocation failed\n");
        stbi_image_free(img);
        return 1;
    }

    clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Processed %d pixels in %f seconds\n", width * height, time_taken);

    if (!stbi_write_png("output/gaussian.png", width, height, 1, gray_img, width)) {
        printf("Failed to write image\n");
    } else {
        printf("Grayscale image saved to gray_image.png\n");
    }

    stbi_image_free(img);
    free(gray_img);

    return 0;
}
