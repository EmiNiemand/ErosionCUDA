#include <cstdint>
#include <iostream>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "thirdparty/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "thirdparty/stb_image_write.h"

#define CHANNEL_NUMBER 3

int main(int, char**)
{
    int width, height, bpp;

    std::filesystem::path path = std::filesystem::current_path().parent_path();
    path /= "image.png";

    unsigned char* rgb_image = stbi_load(path.string().c_str(), &width, &height, &bpp, CHANNEL_NUMBER);


    std::cout << "Loaded image" << std::endl;

    for (int i = 0; i < width * height * CHANNEL_NUMBER; i++) {
        rgb_image[i] = 255 - rgb_image[i];
    }

    stbi_write_png(path.string().c_str(), width, height, CHANNEL_NUMBER, rgb_image, width * CHANNEL_NUMBER);

    stbi_image_free(rgb_image);

    std::cout << "Save image" << std::endl;
    return 0;
}