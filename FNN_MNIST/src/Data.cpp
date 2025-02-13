#include "Data.hpp"

Data::Data(size_t rows, size_t cols)
    : values(rows * cols, 0.0), rows(rows), cols(cols) {}

Data::Data() : rows(0), cols(0) {}

double Data::get(size_t row, size_t col) const
{
    if (row >= rows || col >= cols)
        throw std::out_of_range("Index out of range");
    return values[row * cols + col];
}

void Data::set(size_t row, size_t col, double value)
{
    if (row >= rows || col >= cols)
        throw std::out_of_range("Index out of range");
    values[row * cols + col] = value;
}

double &Data::operator()(size_t row, size_t col)
{
    if (row >= rows || col >= cols)
        throw std::out_of_range("Index out of range");
    return values[row * cols + col];
}

const double &Data::operator()(size_t row, size_t col) const
{
    if (row >= rows || col >= cols)
        throw std::out_of_range("Index out of range");
    return values[row * cols + col];
}

size_t Data::getRows() const
{
    return rows;
}

size_t Data::getCols() const
{
    return cols;
}
