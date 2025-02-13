#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>

class Layer
{
public:
    virtual ~Layer() {}
    // Forward pass: process input and produce output.
    virtual std::vector<double> forward(const std::vector<double> &input) = 0;
    // Backward pass: receive gradient from next layer and return gradient for previous layer.
    virtual std::vector<double> backward(const std::vector<double> &grad_output, double learning_rate) = 0;
};

#endif // LAYER_HPP
