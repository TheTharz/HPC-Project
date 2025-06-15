#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <omp.h>

uint8_t* convert_to_grayscale(unsigned char *img, int width, int height) {
    uint8_t *gray_img = malloc(width * height);
    if (!gray_img) {
        return NULL;
    }

    #pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        int idx = i * 3;
        gray_img[i] = (uint8_t)(0.3 * img[idx] + 0.59 * img[idx + 1] + 0.11 * img[idx + 2]);
    }

    return gray_img;
}

int main() {
    //image loading
    int width, height, channels;
    unsigned char *img = stbi_load("../images/test.jpg", &width, &height, &channels, 3);
    if (!img) {
        printf("Failed to load image\n");
        return 1;
    }

    double start_time = omp_get_wtime();

    uint8_t *gray_img = convert_to_grayscale(img, width, height);

    double end_time = omp_get_wtime();

    if (!gray_img) {
      printf("Memory allocation failed\n");
      stbi_image_free(img);
      return 1;
    }

    double time_taken = end_time - start_time;
    printf("Processed %d pixels in %f seconds\n", width * height, time_taken);

    //saving the output
    if (!stbi_write_png("output/gray_image.png", width, height, 1, gray_img, width)) {
        printf("Failed to write image\n");
    } else {
        printf("Grayscale image saved to gray_image.png\n");
    }

    stbi_image_free(img);
    free(gray_img);

    return 0;
}
