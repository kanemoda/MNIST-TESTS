#include "DataHandler.hpp"
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <cstdint>

// Helper function to read a 32-bit unsigned integer in big-endian order
static uint32_t readBigEndianUint32(std::ifstream &stream)
{
    unsigned char bytes[4];
    stream.read(reinterpret_cast<char *>(bytes), 4);
    if (!stream)
    {
        throw std::runtime_error("Failed to read 4 bytes from stream.");
    }
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) |
           static_cast<uint32_t>(bytes[3]);
}

std::vector<Data> DataHandler::loadMNISTImages(const std::string &filePath, bool normalize)
{
    std::vector<Data> images;
    std::ifstream inFile(filePath, std::ios::binary);
    if (!inFile)
    {
        throw std::runtime_error("Cannot open image file: " + filePath);
    }

    // MNIST image file header consists of:
    // [Magic Number (4 bytes)] [Number of Images (4 bytes)]
    // [Number of Rows (4 bytes)]   [Number of Columns (4 bytes)]
    uint32_t magic = readBigEndianUint32(inFile);
    if (magic != 2051)
    {
        std::stringstream ss;
        ss << "Invalid MNIST image file magic number: " << magic;
        throw std::runtime_error(ss.str());
    }

    uint32_t numImages = readBigEndianUint32(inFile);
    uint32_t numRows = readBigEndianUint32(inFile);
    uint32_t numCols = readBigEndianUint32(inFile);

    // Read each image
    for (uint32_t i = 0; i < numImages; i++)
    {
        Data img(numRows, numCols);
        for (uint32_t row = 0; row < numRows; row++)
        {
            for (uint32_t col = 0; col < numCols; col++)
            {
                unsigned char pixel = 0;
                inFile.read(reinterpret_cast<char *>(&pixel), sizeof(pixel));
                if (!inFile)
                {
                    throw std::runtime_error("Error reading pixel data for image " + std::to_string(i));
                }
                double value = static_cast<double>(pixel);
                if (normalize)
                {
                    value /= 255.0; // Normalize pixel value to range [0,1]
                }
                img.set(row, col, value);
            }
        }
        images.push_back(img);
    }
    return images;
}

std::vector<unsigned char> DataHandler::loadMNISTLabels(const std::string &filePath)
{
    std::vector<unsigned char> labels;
    std::ifstream inFile(filePath, std::ios::binary);
    if (!inFile)
    {
        throw std::runtime_error("Cannot open label file: " + filePath);
    }

    // MNIST label file header consists of:
    // [Magic Number (4 bytes)] [Number of Labels (4 bytes)]
    uint32_t magic = readBigEndianUint32(inFile);
    if (magic != 2049)
    {
        std::stringstream ss;
        ss << "Invalid MNIST label file magic number: " << magic;
        throw std::runtime_error(ss.str());
    }

    uint32_t numLabels = readBigEndianUint32(inFile);
    labels.resize(numLabels);

    // Read all label bytes in one go
    inFile.read(reinterpret_cast<char *>(labels.data()), numLabels * sizeof(unsigned char));
    if (!inFile)
    {
        throw std::runtime_error("Error reading label data from file: " + filePath);
    }

    return labels;
}
