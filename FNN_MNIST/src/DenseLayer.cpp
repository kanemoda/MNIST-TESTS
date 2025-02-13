#include "DenseLayer.hpp"
#include <cstdlib>
#include <sstream>
#include <Eigen/Dense>
#include <omp.h>

DenseLayer::DenseLayer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size)
{
    weights.resize(output_size, std::vector<double>(input_size));
    biases.resize(output_size, 0.0);
    for (int i = 0; i < output_size; i++)
    {
        for (int j = 0; j < input_size; j++)
        {
            weights[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
}

std::vector<double> DenseLayer::forward(const std::vector<double> &input)
{
    input_cache = input;

    // Option A: Use a simple loop with OpenMP parallelization.
    /*
    std::vector<double> output(output_size, 0.0);
#pragma omp parallel for
    for (int i = 0; i < output_size; i++)
    {
        double sum = biases[i];
        for (int j = 0; j < input_size; j++)
        {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum;
    }
    return output;
    */

    // Option B: Use Eigen for matrix multiplication (uncomment to use).

    Eigen::Map<const Eigen::VectorXd> eigenInput(input.data(), input.size());
    Eigen::MatrixXd eigenWeights(output_size, input_size);
    for (int i = 0; i < output_size; i++)
    {
        for (int j = 0; j < input_size; j++)
        {
            eigenWeights(i, j) = weights[i][j];
        }
    }
    Eigen::VectorXd eigenOutput = eigenWeights * eigenInput;
    for (int i = 0; i < output_size; i++)
    {
        eigenOutput(i) += biases[i];
    }
    std::vector<double> output(eigenOutput.data(), eigenOutput.data() + eigenOutput.size());
    return output;
}

std::vector<double> DenseLayer::backward(const std::vector<double> &grad_output, double learning_rate)
{
    std::vector<double> grad_input(input_size, 0.0);

// Simple loop with OpenMP parallelization.
#pragma omp parallel for
    for (int i = 0; i < output_size; i++)
    {
        for (int j = 0; j < input_size; j++)
        {
            double grad_weight = grad_output[i] * input_cache[j];
// Atomically update weights to avoid race conditions.
#pragma omp atomic
            weights[i][j] -= learning_rate * grad_weight;
#pragma omp atomic
            grad_input[j] += weights[i][j] * grad_output[i];
        }
#pragma omp atomic
        biases[i] -= learning_rate * grad_output[i];
    }
    return grad_input;
}
