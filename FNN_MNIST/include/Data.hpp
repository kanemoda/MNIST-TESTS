#ifndef DATA_HPP
#define DATA_HPP

#include <vector>
#include <stdexcept>
#include <cstddef>

class Data
{
private:
    std::vector<double> values; // Holds the pixel data (or any data)
    size_t rows;
    size_t cols;

public:
    // Constructor that creates a data matrix with given rows and columns
    Data(size_t rows, size_t cols);

    // Default constructor (creates an empty Data object)
    Data();

    // Get the value at (row, col)
    double get(size_t row, size_t col) const;

    // Set the value at (row, col)
    void set(size_t row, size_t col, double value);

    // Overloaded operator() for element access (non-const)
    double &operator()(size_t row, size_t col);

    // Overloaded operator() for element access (const)
    const double &operator()(size_t row, size_t col) const;

    // Accessor functions for dimensions
    size_t getRows() const;
    size_t getCols() const;
};

#endif // DATA_HPP
