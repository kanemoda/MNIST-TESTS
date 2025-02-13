#include "DataHandler.hpp"
#include "Algorithm.hpp"
#include <iostream>

int main()
{
    // Create a DataHandler object
    DataHandler dataHandler;

    // File paths for the MNIST training data (adjust if you want to use test files)
    std::string trainImages = "../mnist_data/train-images-idx3-ubyte";
    std::string trainLabels = "../mnist_data/train-labels-idx1-ubyte";

    // Load the MNIST data
    if (!dataHandler.loadData(trainImages, trainLabels))
    {
        std::cerr << "Error: Could not load MNIST data." << std::endl;
        return 1;
    }

    // Create an Algorithm object and run training/testing
    Algorithm algo;
    // This method will perform an 80% training / 20% testing split and print the accuracy
    algo.trainAndTest(dataHandler, 0.8f);

    return 0;
}
