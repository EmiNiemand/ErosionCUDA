#include <iostream>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "thirdparty/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "thirdparty/stb_image_write.h"

#define CHANNEL_NUMBER 3


unsigned int* RBGToBW (unsigned char* image, int arraySize) {
    unsigned int* output = new unsigned int[arraySize / CHANNEL_NUMBER];

    int sum = 0;

    for (int i = 0; i < arraySize; i++) {
        sum += image[i];
        if ((i + 1) % CHANNEL_NUMBER == 0) {
            if (sum > 0) output[i / CHANNEL_NUMBER] = 1;
            else output[i / CHANNEL_NUMBER] = 0;
            sum = 0;
        }
    }

    return output;
}

unsigned char* BWToRGB (unsigned int* image, int arraySize) {
    unsigned char* output = new unsigned char[arraySize * CHANNEL_NUMBER];

    for (int i = 0; i < arraySize; i++) {
        for (int j = 0; j < CHANNEL_NUMBER; j++) {
            output[i * CHANNEL_NUMBER + j] = 255 * image[i];
        }
    }

    return output;
}

__global__ void erosion_kernel(unsigned int* input, unsigned int* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int output_value = input[y * width + x]; // get center pixel value

        if (y > 0 && input[(y - 1) * width + x] < output_value) {
            output_value = input[(y - 1) * width + x]; // check top pixel
        }

        if (y < height - 1 && input[(y + 1) * width + x] < output_value) {
            output_value = input[(y + 1) * width + x]; // check bottom pixel
        }

        if (x > 0 && input[y * width + (x - 1)] < output_value) {
            output_value = input[y * width + (x - 1)]; // check left pixel
        }

        if (x < width - 1 && input[y * width + (x + 1)] < output_value) {
            output_value = input[y * width + (x + 1)]; // check right pixel
        }

        output[y * width + x] = output_value; // write output value to output data array
    }
}

int main() {
    int width, height, bpp;

    int erosionDepth = 1;

    std::cout << "Erosion depth / level (uint from 1 to uint max): ";
    std::cin >> erosionDepth;

    std::filesystem::path path = std::filesystem::current_path().parent_path();
    path /= "image.png";

    unsigned char* image = stbi_load(path.string().c_str(), &width, &height, &bpp, CHANNEL_NUMBER);
    std::cout << "Loaded image" << std::endl;

    unsigned int* host_input = RBGToBW(image, width * height * CHANNEL_NUMBER);
    unsigned int* host_output = (unsigned int*)malloc(width * height * sizeof(unsigned int));
    unsigned int* device_input;
    unsigned int* device_output;

    cudaMalloc((void**)&device_input, width * height * sizeof(unsigned int));
    cudaMalloc((void**)&device_output, width * height * sizeof(unsigned int));

    cudaMemcpy(device_input, host_input, width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 block_size(32, 32);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    for (int i = 0; i < erosionDepth; i++) {
        erosion_kernel<<<grid_size, block_size>>>(device_input, device_output, width, height);
        cudaMemcpy(device_input, device_output, width * height * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(host_output, device_output, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    image = BWToRGB(host_output, width * height);

    stbi_write_png(path.string().c_str(), width, height, CHANNEL_NUMBER, image, width * CHANNEL_NUMBER);

    cudaFree(device_input);
    cudaFree(device_output);

    free(host_input);
    free(host_output);

    delete[] image;

    std::cout << "Save image" << std::endl;
    return 0;
}
