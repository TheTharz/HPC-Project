#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// Function to convert RGB image to grayscale
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
    int width, height, channels;
    unsigned char *img = stbi_load("output/sobel.png", &width, &height, &channels, 3);
    if (!img) {
        printf("Failed to load image\n");
        return 1;
    }

    clock_t start_time = clock();

    uint8_t *gray_img = convert_to_grayscale(img, width, height);
    if (!gray_img) {
        printf("Memory allocation failed\n");
        stbi_image_free(img);
        return 1;
    }

    clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Processed %d pixels in %f seconds\n", width * height, time_taken);

    if (!stbi_write_png("output/sobel_gray_image.png", width, height, 1, gray_img, width)) {
        printf("Failed to write image\n");
    } else {
        printf("Grayscale image saved to gray_image.png\n");
    }

    stbi_image_free(img);
    free(gray_img);

    return 0;
}
