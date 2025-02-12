#ifndef PCA_HPP
#define PCA_HPP

#include <vector>
#include <cstddef>

class PCA
{
public:
    // Number of principal components to retain.
    size_t n_components;
    // Mean vector computed from training data.
    std::vector<double> mean;
    // Principal components (each is a vector of length = original feature dimension).
    std::vector<std::vector<double>> components;

    // Constructor.
    PCA(size_t n_components);

    // Fit PCA on the given data.
    // 'data' is a matrix with each row as a sample.
    // max_iterations and tol control the convergence of the eigenvector estimation.
    void fit(const std::vector<std::vector<double>> &data, size_t max_iterations = 1000, double tol = 1e-6);

    // Transform data into the reduced space.
    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>> &data) const;
};

#endif // PCA_HPP
