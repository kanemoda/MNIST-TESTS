#include "NeuralNetwork.hpp"
#include <cassert>

NeuralNetwork::NeuralNetwork() {}

void NeuralNetwork::addLayer(std::unique_ptr<Layer> layer)
{
    layers.push_back(std::move(layer));
}

std::vector<double> NeuralNetwork::predict(const std::vector<double> &input)
{
    std::vector<double> output = input;
    for (auto &layer : layers)
    {
        output = layer->forward(output);
    }
    return output;
}

void NeuralNetwork::train(const std::vector<double> &input, const std::vector<double> &target, double learning_rate)
{
    // Forward pass and cache outputs.
    std::vector<double> output = input;
    for (auto &layer : layers)
    {
        output = layer->forward(output);
    }

    // Compute simple MSE loss derivative.
    std::vector<double> grad(output.size());
    for (size_t i = 0; i < output.size(); i++)
    {
        grad[i] = 2 * (output[i] - target[i]);
    }

    // Backward pass.
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        grad = layers[i]->backward(grad, learning_rate);
    }
}
