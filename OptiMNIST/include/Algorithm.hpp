#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "DataHandler.hpp"

class Algorithm
{
public:
    // Performs a simple 1-NN classification with an 80/20 train/test split.
    // Prints the accuracy after testing.
    void trainAndTest(const DataHandler &handler, float trainRatio = 0.8f);
};

#endif // ALGORITHM_H
