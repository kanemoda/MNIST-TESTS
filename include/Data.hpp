#ifndef DATA_H
#define DATA_H

#include <vector>
#include <string>
#include <cstdint>

class Data
{
public:
    // Each image is stored as a vector of 8-bit pixel values.
    std::vector<std::vector<uint8_t>> images;
    // Labels for each image.
    std::vector<uint8_t> labels;

    // Loads images from an IDX file.
    bool loadImages(const std::string &filename);
    // Loads labels from an IDX file.
    bool loadLabels(const std::string &filename);
};

#endif // DATA_H
