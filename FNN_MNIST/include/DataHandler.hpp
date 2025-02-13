#ifndef DATA_HANDLER_HPP
#define DATA_HANDLER_HPP

#include <vector>
#include <string>
#include "Data.hpp"

class DataHandler
{
public:
    // Loads MNIST images from the given file path.
    // If normalize is true, pixel values are scaled to [0,1].
    std::vector<Data> loadMNISTImages(const std::string &filePath, bool normalize = true);

    // Loads MNIST labels from the given file path.
    std::vector<unsigned char> loadMNISTLabels(const std::string &filePath);
};

#endif // DATA_HANDLER_HPP
