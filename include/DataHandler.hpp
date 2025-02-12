#ifndef DATAHANDLER_H
#define DATAHANDLER_H

#include "Data.hpp"
#include <string>

class DataHandler
{
public:
    Data mnistData;
    int imageRows;
    int imageCols;

    // Default image size for MNIST is 28x28.
    DataHandler(int rows = 28, int cols = 28) : imageRows(rows), imageCols(cols) {}

    // Loads both images and labels.
    bool loadData(const std::string &imageFile, const std::string &labelFile);
    // Prints an image (with a simple threshold to binarize).
    void printImage(int index, uint8_t threshold = 128);
};

#endif // DATAHANDLER_H
