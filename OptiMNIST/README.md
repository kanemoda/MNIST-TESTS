# OptiMNIST

**OptiMNIST** is a fast and optimized C++ project for handwritten digit recognition on the MNIST dataset. It leverages PCA-based dimensionality reduction and a SIMD-accelerated 1-Nearest Neighbor (1-NN) classifier, combined with OpenMP parallelization, to achieve approximately 97.7% accuracy with minimal runtime.

## Overview

OptiMNIST employs several modern optimization techniques:
- **PCA Dimensionality Reduction:** Reduces the original 784-dimensional pixel space to a lower-dimensional space (e.g., 50 dimensions) while retaining the most critical features.
- **SIMD-Accelerated 1-NN Classification:** Uses AVX intrinsics for fast Euclidean distance computations.
- **OpenMP Parallelization:** Distributes the workload across multiple threads for efficient processing.
- **Lightweight and Efficient:** Achieves competitive accuracy (≈97.7%) without the overhead of deep neural networks.

## Features

- **Efficient Preprocessing:** Scales input pixel values to the range [0,1].
- **Dimensionality Reduction:** Uses PCA to reduce feature space, accelerating subsequent classification.
- **Optimized Distance Computation:** AVX-based SIMD operations significantly speed up the distance calculations.
- **Parallel Classification:** OpenMP is used to parallelize the 1-NN classification, with progress updates every 2 seconds.
- **High Accuracy:** Achieves around 97.7% accuracy on the MNIST test set.

## File Structure
```
.
├── include
│   ├── Algorithm.hpp
│   ├── DataHandler.hpp
│   ├── Data.hpp
│   ├── LinearAlgebra.hpp
│   └── PCA.hpp
├── mnist_data
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
│   ├── train-images-idx3-ubyte
│   └── train-labels-idx1-ubyte
├── obj
├── src
│   ├── Algorithm.cpp
│   ├── Data.cpp
│   ├── DataHandler.cpp
│   ├── LinearAlgebra.cpp
│   ├── main.cpp
│   └── PCA.cpp
└── Makefile
```

## Prerequisites

- **Compiler:** A C++ compiler with C++11 support.
- **CPU Support:** AVX instructions and OpenMP support (GCC recommended).
- **MNIST Dataset:** Download the MNIST files (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, and `t10k-labels.idx1-ubyte`) and place them in the mnist_data folder.

## Installation & Usage

1. **Clone the Repository:**
```bash
    git clone <repository-url>
    cd OptiMNIST
```
2. **Place MNIST Data:**
   Ensure the MNIST files are placed in the `mnist_data` directory.

3. **Build the Project:**
   ```bash
   make

4. **Run the Project:**
   ```bash
   make run

During execution, progress updates are printed every 2 seconds. After processing, the program outputs the total number of test samples, the number of correct predictions, and the overall accuracy.

## Results

- **Test Accuracy:** ~97.7%
- **Fast Inference:** Thanks to SIMD acceleration and parallel processing, classification is performed very efficiently.

