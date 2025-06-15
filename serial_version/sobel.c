#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// Apply Laplacian filter to a grayscale image
uint8_t* apply_sobel_filter(unsigned char *img, int width, int height, int channels) {
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
    
    uint8_t *output = malloc(width * height * channels);
    if (!output) return NULL;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sumX[3] = { 0, 0, 0 };
            int sumY[3] = { 0, 0, 0 };

            // Apply 3x3 Sobel kernel with zero-padding
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int idx = (ny * width + nx) * channels;
                        for (int c = 0; c < channels; c++) {
                            int pixel = img[idx + c];
                            sumX[c] += pixel * Gx[ky + 1][kx + 1];
                            sumY[c] += pixel * Gy[ky + 1][kx + 1];
                        }
                    }
                }
            }

            // Calculate gradient magnitude
            int magnitude[3];
            for (int c = 0; c < channels; c++) {
                magnitude[c] = (int) sqrt(sumX[c] * sumX[c] + sumY[c] * sumY[c]);
            }

            // Clamp to [0, 255]
            for (int c = 0; c < channels; c++) {
                if (magnitude[c] > 255) magnitude[c] = 255;
                if (magnitude[c] < 0) magnitude[c] = 0;
            }

            int idx = (y * width + x) * channels;
            for (int c = 0; c < channels; c++) {
                output[idx + c] = (uint8_t)magnitude[c];
            }
        }
    }

    return output;
}

int main() {
    int width, height, channels;
    unsigned char *img = stbi_load("../images/test.jpg", &width, &height, &channels, 3);
    if (!img) {
        printf("Failed to load image\n");
        return 1;
    }

    clock_t start_time = clock();

    uint8_t *output = apply_sobel_filter(img, width, height, channels);
    if (!output) {
        printf("Memory allocation failed\n");
        stbi_image_free(img);
        return 1;
    }

    clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Processed %d pixels in %f seconds\n", width * height, time_taken);

    if (!stbi_write_png("output/sobel.png", width, height, channels, output, width * channels)) {
        printf("Failed to write image\n");
    } else {
        printf("Grayscale image saved to gray_image.png\n");
    }

    stbi_image_free(img);
    free(output);

    return 0;
}

