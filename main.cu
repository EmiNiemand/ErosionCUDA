#include <iostream>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "thirdparty/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "thirdparty/stb_image_write.h"

#define CHANNEL_NUMBER 3


unsigned int* rgb_to_bw (const unsigned char* image, int array_size) {
    auto* output = new unsigned int[array_size / CHANNEL_NUMBER];

    int color_sum = 0;

    for (int i = 0; i < array_size; i++) {
        color_sum += image[i];
        if ((i + 1) % CHANNEL_NUMBER == 0) {
            if (color_sum > 0) output[i / CHANNEL_NUMBER] = 1;
            else output[i / CHANNEL_NUMBER] = 0;
            color_sum = 0;
        }
    }

    return output;
}

unsigned char* bw_to_rgb (const unsigned int* image, int array_size) {
    auto* output = new unsigned char[array_size * CHANNEL_NUMBER];

    for (int i = 0; i < array_size; i++) {
        for (int j = 0; j < CHANNEL_NUMBER; j++) {
            output[i * CHANNEL_NUMBER + j] = 255 * image[i];
        }
    }

    return output;
}

__global__ void erosion_kernel(unsigned int* image, int width, int height) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        auto output_value = image[y * width + x]; // get center pixel value

        if (y > 0 && image[(y - 1) * width + x] < output_value) {
            output_value = image[(y - 1) * width + x]; // check top pixel
        }

        if (y < height - 1 && image[(y + 1) * width + x] < output_value) {
            output_value = image[(y + 1) * width + x]; // check bottom pixel
        }

        if (x > 0 && image[y * width + (x - 1)] < output_value) {
            output_value = image[y * width + (x - 1)]; // check left pixel
        }

        if (x < width - 1 && image[y * width + (x + 1)] < output_value) {
            output_value = image[y * width + (x + 1)]; // check right pixel
        }

        image[y * width + x] = output_value; // write output value to output data array
    }
}

int main() {
    int width, height, chif, erosion_level;

    std::cout << "Choose erosion level (uint from 1 to uint max): ";
    std::cin >> erosion_level;

    std::filesystem::path input_path = std::filesystem::current_path().parent_path();
    input_path /= "image.png";

    std::filesystem::path output_path = std::filesystem::current_path().parent_path();
    output_path /= "eroded_image.png";

    unsigned char* image = stbi_load(input_path.string().c_str(), &width, &height, &chif, CHANNEL_NUMBER);
    std::cout << "Loaded image" << std::endl;

    unsigned int* host_input = rgb_to_bw(image, width * height * CHANNEL_NUMBER);
    auto* host_output = (unsigned int*)malloc(width * height * sizeof(unsigned int));
    unsigned int* device;

    cudaMalloc((void**)&device, width * height * sizeof(unsigned int));

    cudaMemcpy(device, host_input, width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 block_size(32, 32);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    for (int i = 0; i < erosion_level; i++) {
        erosion_kernel<<<grid_size, block_size>>>(device, width, height);
    }

    cudaMemcpy(host_output, device, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    image = bw_to_rgb(host_output, width * height);

    stbi_write_png(output_path.string().c_str(), width, height, CHANNEL_NUMBER, image, width * CHANNEL_NUMBER);
    std::cout << "Save image" << std::endl;

    cudaFree(device);

    free(host_input);
    free(host_output);

    delete[] image;

    return 0;
}
