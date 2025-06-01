#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>

int main() {
    int width, height, channels;
    unsigned char *img = stbi_load("images/test.jpg", &width, &height, &channels, 0);

    if (img == NULL) {
        printf("Failed to load image\n");
        return 1;
    }

    size_t img_size = width * height * channels;
    unsigned char *gray_img = malloc(width * height);

    if (gray_img == NULL) {
        printf("Failed to allocate memory\n");
        stbi_image_free(img);
        return 1;
    }

    for (int i = 0; i < width * height; i++) {
        int r = img[i * channels + 0];
        int g = img[i * channels + 1];
        int b = img[i * channels + 2];
        unsigned char gray = (r * 0.3) + (g * 0.59) + (b * 0.11);
        gray_img[i] = gray;
    }

    stbi_write_png("output/gray_image.png", width, height, 1, gray_img, width);

    stbi_image_free(img);
    free(gray_img);

    printf("Grayscale image saved to output/gray_image.png\n");

    return 0;
}
