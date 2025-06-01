#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// Apply Laplacian filter to a grayscale image
uint8_t* apply_laplacian_filter(uint8_t *gray_img, int width, int height) {
    int kernel[3][3] = {
        {  0, -1,  0 },
        { -1,  4, -1 },
        {  0, -1,  0 }
    };

    uint8_t *output = malloc(width * height);
    if (!output) return NULL;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;

                    // Zero padding: treat out-of-bounds as 0
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

int main() {
    int width, height, channels;
    unsigned char *img = stbi_load("output/gray_image.png", &width, &height, &channels, 3);
    if (!img) {
        printf("Failed to load image\n");
        return 1;
    }

    clock_t start_time = clock();

    uint8_t *gray_img = apply_laplacian_filter(img, width, height);
    if (!gray_img) {
        printf("Memory allocation failed\n");
        stbi_image_free(img);
        return 1;
    }

    clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Processed %d pixels in %f seconds\n", width * height, time_taken);

    if (!stbi_write_png("output/laplacian.png", width, height, 1, gray_img, width)) {
        printf("Failed to write image\n");
    } else {
        printf("Grayscale image saved to gray_image.png\n");
    }

    stbi_image_free(img);
    free(gray_img);

    return 0;
}
