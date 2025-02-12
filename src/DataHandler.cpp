#include "DataHandler.hpp"
#include <iostream>

bool DataHandler::loadData(const std::string &imageFile, const std::string &labelFile)
{
    bool imagesLoaded = mnistData.loadImages(imageFile);
    bool labelsLoaded = mnistData.loadLabels(labelFile);
    return imagesLoaded && labelsLoaded;
}

void DataHandler::printImage(int index, uint8_t threshold)
{
    if (index < 0 || index >= static_cast<int>(mnistData.images.size()))
    {
        std::cerr << "Index out of range." << std::endl;
        return;
    }
    const auto &image = mnistData.images[index];
    for (int i = 0; i < imageRows; i++)
    {
        for (int j = 0; j < imageCols; j++)
        {
            uint8_t pixel = image[i * imageCols + j];
            std::cout << (pixel > threshold ? "X" : "-");
        }
        std::cout << std::endl;
    }
    std::cout << "Label: " << static_cast<int>(mnistData.labels[index]) << std::endl;
}
