#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include <vector>

namespace la
{

    // Computes the dot product of two vectors.
    double dot(const std::vector<double> &a, const std::vector<double> &b);

    // Computes the L2 norm of a vector.
    double norm(const std::vector<double> &v);

    // Normalizes a vector in place.
    void normalize(std::vector<double> &v);

    // Multiplies a matrix by a vector (each row of the matrix dotted with the vector).
    std::vector<double> multiply(const std::vector<std::vector<double>> &A, const std::vector<double> &x);

} // namespace la

#endif // LINEAR_ALGEBRA_HPP
