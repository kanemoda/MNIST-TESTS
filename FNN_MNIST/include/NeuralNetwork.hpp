#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "Layer.hpp"
#include <vector>
#include <memory>

class NeuralNetwork
{
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    NeuralNetwork();
    void addLayer(std::unique_ptr<Layer> layer);
    std::vector<double> predict(const std::vector<double> &input);
    void train(const std::vector<double> &input, const std::vector<double> &target, double learning_rate);
};

#endif // NEURAL_NETWORK_HPP
