#include "PCA.hpp"
#include "LinearAlgebra.hpp" // Our custom linear algebra functions.
#include <cmath>
#include <limits>
#include <iostream>
#include <cassert>
#include <algorithm>

// Helper: perform power iteration to compute the dominant eigenvector and eigenvalue
// for a symmetric matrix.
static void power_iteration(const std::vector<std::vector<double>> &matrix,
                            std::vector<double> &eigenvector,
                            double &eigenvalue,
                            size_t max_iterations,
                            double tol)
{
    size_t n = matrix.size();
    eigenvector.assign(n, 1.0); // Initialize with ones.
    la::normalize(eigenvector);

    std::vector<double> new_vec(n, 0.0);
    for (size_t iter = 0; iter < max_iterations; iter++)
    {
        // Multiply matrix by current eigenvector estimate.
        for (size_t i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (size_t j = 0; j < n; j++)
            {
                sum += matrix[i][j] * eigenvector[j];
            }
            new_vec[i] = sum;
        }
        double new_norm = la::norm(new_vec);
        if (new_norm == 0)
            break;
        for (size_t i = 0; i < n; i++)
        {
            new_vec[i] /= new_norm;
        }
        // Check convergence.
        double diff = 0.0;
        for (size_t i = 0; i < n; i++)
        {
            diff += std::abs(new_vec[i] - eigenvector[i]);
        }
        eigenvector = new_vec;
        if (diff < tol)
            break;
    }
    // Compute eigenvalue using the Rayleigh quotient.
    eigenvalue = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        double sum = 0.0;
        for (size_t j = 0; j < n; j++)
        {
            sum += matrix[i][j] * eigenvector[j];
        }
        eigenvalue += eigenvector[i] * sum;
    }
}

PCA::PCA(size_t n_components_) : n_components(n_components_) {}

void PCA::fit(const std::vector<std::vector<double>> &data, size_t max_iterations, double tol)
{
    size_t n_samples = data.size();
    size_t n_features = data[0].size();
    // Compute the mean vector.
    mean.assign(n_features, 0.0);
    for (const auto &sample : data)
    {
        for (size_t j = 0; j < n_features; j++)
        {
            mean[j] += sample[j];
        }
    }
    for (size_t j = 0; j < n_features; j++)
    {
        mean[j] /= n_samples;
    }
    // Center the data.
    std::vector<std::vector<double>> centered = data;
    for (auto &sample : centered)
    {
        for (size_t j = 0; j < n_features; j++)
        {
            sample[j] -= mean[j];
        }
    }
    // Compute the covariance matrix.
    std::vector<std::vector<double>> cov(n_features, std::vector<double>(n_features, 0.0));
    for (size_t i = 0; i < n_samples; i++)
    {
        for (size_t j = 0; j < n_features; j++)
        {
            for (size_t k = 0; k < n_features; k++)
            {
                cov[j][k] += centered[i][j] * centered[i][k];
            }
        }
    }
    for (size_t j = 0; j < n_features; j++)
    {
        for (size_t k = 0; k < n_features; k++)
        {
            cov[j][k] /= (n_samples - 1);
        }
    }

    components.clear();
    // Use power iteration with deflation to compute the first n_components eigenvectors.
    std::vector<std::vector<double>> cov_copy = cov; // copy for deflation
    for (size_t comp = 0; comp < n_components; comp++)
    {
        std::vector<double> eigenvector(n_features, 0.0);
        double eigenvalue = 0.0;
        power_iteration(cov_copy, eigenvector, eigenvalue, max_iterations, tol);
        components.push_back(eigenvector);
        // Deflate the covariance matrix.
        for (size_t i = 0; i < n_features; i++)
        {
            for (size_t j = 0; j < n_features; j++)
            {
                cov_copy[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }
}

std::vector<std::vector<double>> PCA::transform(const std::vector<std::vector<double>> &data) const
{
    size_t n_samples = data.size();
    size_t n_features = data[0].size();
    std::vector<std::vector<double>> transformed(n_samples, std::vector<double>(n_components, 0.0));
    // For each sample: center it and project onto each principal component.
    for (size_t i = 0; i < n_samples; i++)
    {
        std::vector<double> centered = data[i];
        for (size_t j = 0; j < n_features; j++)
        {
            centered[j] -= mean[j];
        }
        for (size_t comp = 0; comp < n_components; comp++)
        {
            transformed[i][comp] = la::dot(centered, components[comp]);
        }
    }
    return transformed;
}
