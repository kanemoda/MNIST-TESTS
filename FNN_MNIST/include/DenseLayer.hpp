#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "Layer.hpp"
#include <vector>

class DenseLayer : public Layer
{
private:
    int input_size, output_size;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> input_cache; // Cache the input for use in backpropagation

public:
    DenseLayer(int input_size, int output_size);
    virtual std::vector<double> forward(const std::vector<double> &input) override;
    virtual std::vector<double> backward(const std::vector<double> &grad_output, double learning_rate) override;
};

#endif // DENSE_LAYER_HPP
