#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include "Layer.hpp"
#include <vector>

enum class ActivationType
{
    RELU,
    SIGMOID
};

class ActivationLayer : public Layer
{
private:
    ActivationType type;
    std::vector<double> input_cache; // Cache the input

public:
    ActivationLayer(ActivationType type);
    virtual std::vector<double> forward(const std::vector<double> &input) override;
    virtual std::vector<double> backward(const std::vector<double> &grad_output, double learning_rate) override;
};

#endif // ACTIVATION_LAYER_HPP
