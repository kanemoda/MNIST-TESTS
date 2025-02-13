#include "ActivationLayer.hpp"
#include <cmath>

ActivationLayer::ActivationLayer(ActivationType type) : type(type) {}

std::vector<double> ActivationLayer::forward(const std::vector<double> &input)
{
    input_cache = input;
    std::vector<double> output(input.size());
    if (type == ActivationType::RELU)
    {
        for (size_t i = 0; i < input.size(); i++)
        {
            output[i] = input[i] > 0 ? input[i] : 0;
        }
    }
    else if (type == ActivationType::SIGMOID)
    {
        for (size_t i = 0; i < input.size(); i++)
        {
            output[i] = 1.0 / (1.0 + exp(-input[i]));
        }
    }
    return output;
}

std::vector<double> ActivationLayer::backward(const std::vector<double> &grad_output, double /*learning_rate*/)
{
    std::vector<double> grad_input(input_cache.size());
    if (type == ActivationType::RELU)
    {
        for (size_t i = 0; i < input_cache.size(); i++)
        {
            double derivative = input_cache[i] > 0 ? 1.0 : 0.0;
            grad_input[i] = grad_output[i] * derivative;
        }
    }
    else if (type == ActivationType::SIGMOID)
    {
        for (size_t i = 0; i < input_cache.size(); i++)
        {
            double sigmoid_val = 1.0 / (1.0 + exp(-input_cache[i]));
            double derivative = sigmoid_val * (1 - sigmoid_val);
            grad_input[i] = grad_output[i] * derivative;
        }
    }
    return grad_input;
}
