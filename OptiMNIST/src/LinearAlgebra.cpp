#include "LinearAlgebra.hpp"
#include <cmath>

namespace la
{

    double dot(const std::vector<double> &a, const std::vector<double> &b)
    {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }

    double norm(const std::vector<double> &v)
    {
        return std::sqrt(dot(v, v));
    }

    void normalize(std::vector<double> &v)
    {
        double n = norm(v);
        if (n > 0)
        {
            for (auto &x : v)
            {
                x /= n;
            }
        }
    }

    std::vector<double> multiply(const std::vector<std::vector<double>> &A, const std::vector<double> &x)
    {
        std::vector<double> result(A.size(), 0.0);
        for (size_t i = 0; i < A.size(); i++)
        {
            result[i] = dot(A[i], x);
        }
        return result;
    }

} // namespace la
