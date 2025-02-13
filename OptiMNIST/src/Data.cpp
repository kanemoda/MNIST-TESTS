#include "Data.hpp"
#include <fstream>
#include <iostream>
#include <cstdint>

// Helper: converts big-endian to little-endian.
static uint32_t swapEndian(uint32_t val)
{
    return ((val >> 24) & 0xff) |
           ((val << 8) & 0xff0000) |
           ((val >> 8) & 0xff00) |
           ((val << 24) & 0xff000000);
}

bool Data::loadImages(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Unable to open image file: " << filename << std::endl;
        return false;
    }

    uint32_t magic_number = 0;
    uint32_t num_images = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;

    file.read(reinterpret_cast<char *>(&magic_number), 4);
    magic_number = swapEndian(magic_number);
    if (magic_number != 2051)
    { // 2051 is the MNIST magic number for images
        std::cerr << "Invalid MNIST image file!" << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char *>(&num_images), 4);
    num_images = swapEndian(num_images);
    file.read(reinterpret_cast<char *>(&rows), 4);
    rows = swapEndian(rows);
    file.read(reinterpret_cast<char *>(&cols), 4);
    cols = swapEndian(cols);

    images.resize(num_images, std::vector<uint8_t>(rows * cols));
    for (uint32_t i = 0; i < num_images; i++)
    {
        file.read(reinterpret_cast<char *>(images[i].data()), rows * cols);
    }
    file.close();
    return true;
}

bool Data::loadLabels(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Unable to open label file: " << filename << std::endl;
        return false;
    }

    uint32_t magic_number = 0;
    uint32_t num_labels = 0;

    file.read(reinterpret_cast<char *>(&magic_number), 4);
    magic_number = swapEndian(magic_number);
    if (magic_number != 2049)
    { // 2049 is the MNIST magic number for labels
        std::cerr << "Invalid MNIST label file!" << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char *>(&num_labels), 4);
    num_labels = swapEndian(num_labels);

    labels.resize(num_labels);
    file.read(reinterpret_cast<char *>(labels.data()), num_labels);
    file.close();
    return true;
}
