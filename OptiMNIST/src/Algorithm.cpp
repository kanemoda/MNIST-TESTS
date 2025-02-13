#include "Algorithm.hpp"
#include "PCA.hpp"
#include "LinearAlgebra.hpp"
#include "DataHandler.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <thread>
#include <vector>
#include <limits>
#include <chrono>
#include <immintrin.h>
#include <omp.h>

// SIMD-optimized squared Euclidean distance (using AVX; requires -mavx flag).
static double squaredEuclideanDistanceSIMD(const double *a, const double *b, size_t n)
{
    size_t i = 0;
    __m256d sum_vec = _mm256_setzero_pd();
    for (; i + 3 < n; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d diff = _mm256_sub_pd(va, vb);
        __m256d sq = _mm256_mul_pd(diff, diff);
        sum_vec = _mm256_add_pd(sum_vec, sq);
    }
    double sum[4];
    _mm256_storeu_pd(sum, sum_vec);
    double total = sum[0] + sum[1] + sum[2] + sum[3];
    for (; i < n; i++)
    {
        double d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

void Algorithm::trainAndTest(const DataHandler &handler, float trainRatio)
{
    // Retrieve data from the DataHandler.
    const auto &images_uint8 = handler.mnistData.images;
    const auto &labels_uint8 = handler.mnistData.labels;
    size_t totalSamples = images_uint8.size();
    if (totalSamples == 0)
    {
        std::cerr << "No data available." << std::endl;
        return;
    }

    int rows = handler.imageRows;
    int cols = handler.imageCols;
    int numPixels = rows * cols;

    // Convert images from uint8_t to double scaled to [0, 1].
    std::vector<std::vector<double>> images_double(totalSamples, std::vector<double>(numPixels, 0.0));
    for (size_t i = 0; i < totalSamples; i++)
    {
        for (size_t j = 0; j < images_uint8[i].size(); j++)
        {
            images_double[i][j] = static_cast<double>(images_uint8[i][j]) / 255.0;
        }
    }

    // Shuffle indices and split into training and testing sets.
    std::vector<size_t> indices(totalSamples);
    for (size_t i = 0; i < totalSamples; i++)
    {
        indices[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    size_t trainSize = static_cast<size_t>(totalSamples * trainRatio);
    std::vector<size_t> trainIndices(indices.begin(), indices.begin() + trainSize);
    std::vector<size_t> testIndices(indices.begin() + trainSize, indices.end());

    // Build training and test data.
    std::vector<std::vector<double>> trainData;
    for (size_t idx : trainIndices)
    {
        trainData.push_back(images_double[idx]);
    }
    std::vector<std::vector<double>> testData;
    for (size_t idx : testIndices)
    {
        testData.push_back(images_double[idx]);
    }
    size_t n_test = testData.size();

    // Perform PCA on the training data.
    size_t n_components = 50; // For example, reduce to 50 dimensions.
    PCA pca(n_components);
    std::cout << "Fitting PCA on training data..." << std::endl;
    pca.fit(trainData);
    std::cout << "Transforming training data with PCA..." << std::endl;
    auto trainTransformed = pca.transform(trainData);
    std::cout << "Transforming test data with PCA..." << std::endl;
    auto testTransformed = pca.transform(testData);

    // Set up progress reporting.
    size_t processed = 0;
    bool done = false;
    std::thread progressThread([&]()
                               {
        while (!done) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            size_t current = processed;
            float percent = (100.0f * current) / static_cast<float>(n_test);
            std::cout << "Progress: " << percent << "% complete (" << current << "/" << n_test << ")" << std::endl;
        } });

    int correct = 0;
// Use OpenMP to parallelize the 1‑NN classification over test samples.
#pragma omp parallel for reduction(+ : correct) schedule(dynamic)
    for (size_t i = 0; i < n_test; i++)
    {
        const auto &testVec = testTransformed[i];
        uint8_t trueLabel = labels_uint8[testIndices[i]];
        double bestDistance = std::numeric_limits<double>::max();
        uint8_t predictedLabel = 0;
        // Compare test sample to each training sample.
        for (size_t j = 0; j < trainTransformed.size(); j++)
        {
            double dist = squaredEuclideanDistanceSIMD(testVec.data(), trainTransformed[j].data(), n_components);
            if (dist < bestDistance)
            {
                bestDistance = dist;
                predictedLabel = labels_uint8[trainIndices[j]];
            }
        }
        if (predictedLabel == trueLabel)
        {
            correct++;
        }
#pragma omp atomic
        processed++;
    }

    done = true;
    progressThread.join();

    float accuracy = static_cast<float>(correct) / static_cast<float>(n_test);
    std::cout << "Total Test Samples: " << n_test << std::endl;
    std::cout << "Correct Predictions: " << correct << std::endl;
    std::cout << "Accuracy: " << accuracy * 100.0f << "%" << std::endl;
}
