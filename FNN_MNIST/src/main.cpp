#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <omp.h> // Include OpenMP header
#include "NeuralNetwork.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "DataHandler.hpp"
#include "Data.hpp"

// Helper: flatten a 2D Data object into a 1D vector.
std::vector<double> flattenData(const Data &img)
{
    std::vector<double> flat;
    size_t rows = img.getRows();
    size_t cols = img.getCols();
    flat.reserve(rows * cols);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            flat.push_back(img.get(i, j));
        }
    }
    return flat;
}

// Helper: return the index of the maximum value (predicted label)
int getPredictedLabel(const std::vector<double> &output)
{
    return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
}

int main()
{
    // Set OpenMP to use 12 threads.
    omp_set_num_threads(12);

    // ----------------------------
    // 1. Build the Neural Network
    // ----------------------------
    NeuralNetwork net;
    net.addLayer(std::make_unique<DenseLayer>(784, 128)); // 784 input neurons for 28x28 images
    net.addLayer(std::make_unique<ActivationLayer>(ActivationType::RELU));
    net.addLayer(std::make_unique<DenseLayer>(128, 10)); // 10 output neurons for 10 classes
    net.addLayer(std::make_unique<ActivationLayer>(ActivationType::SIGMOID));

    // -------------------------------------
    // 2. Load MNIST Data using DataHandler
    // -------------------------------------
    DataHandler dh;
    std::string imagesPath = "../mnist_data/train-images-idx3-ubyte";
    std::string labelsPath = "../mnist_data/train-labels-idx1-ubyte";

    std::vector<Data> allImages = dh.loadMNISTImages(imagesPath, true); // 'true' for normalization
    std::vector<unsigned char> allLabels = dh.loadMNISTLabels(labelsPath);

    if (allImages.empty() || allLabels.empty())
    {
        std::cerr << "Error loading data." << std::endl;
        return 1;
    }

    // --------------------------------------------------------
    // 3. Split the Data: 80% for training, 20% for testing
    // --------------------------------------------------------
    size_t totalSamples = allImages.size();
    size_t trainSamples = static_cast<size_t>(totalSamples * 0.8);
    size_t testSamples = totalSamples - trainSamples;

    std::vector<Data> trainImages(allImages.begin(), allImages.begin() + trainSamples);
    std::vector<unsigned char> trainLabels(allLabels.begin(), allLabels.begin() + trainSamples);

    std::vector<Data> testImages(allImages.begin() + trainSamples, allImages.end());
    std::vector<unsigned char> testLabels(allLabels.begin() + trainSamples, allLabels.end());

    // ---------------------------------------------------
    // 4. Training Phase: Mini-Batch Training Loop
    // ---------------------------------------------------
    const size_t batchSize = 32;
    size_t numBatches = trainSamples / batchSize;
    double learning_rate = 0.01;
    int numEpochs = 20; // For demonstration purposes

    std::cout << "Training on " << trainSamples << " samples in " << numBatches << " batches per epoch." << std::endl;
    for (int epoch = 0; epoch < numEpochs; epoch++)
    {
        // Loop over mini-batches
        for (size_t b = 0; b < numBatches; b++)
        {
            // Process each sample in the mini-batch
            for (size_t i = b * batchSize; i < (b + 1) * batchSize; i++)
            {
                std::vector<double> input = flattenData(trainImages[i]);
                std::vector<double> target(10, 0.0);
                target[trainLabels[i]] = 1.0; // One-hot encode the label
                // Scale the learning rate for the batch (a simple simulation of batching)
                net.train(input, target, learning_rate / batchSize);
            }
        }
        std::cout << "Completed Epoch " << epoch << std::endl;
    }

    // ---------------------------------
    // 5. Testing Phase: Evaluate the Network
    // ---------------------------------
    int correct = 0;
    for (size_t i = 0; i < testSamples; i++)
    {
        std::vector<double> input = flattenData(testImages[i]);
        std::vector<double> output = net.predict(input);
        int predictedLabel = getPredictedLabel(output);
        if (predictedLabel == testLabels[i])
        {
            correct++;
        }
    }

    double accuracy = (static_cast<double>(correct) / testSamples) * 100.0;
    std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;

    return 0;
}
