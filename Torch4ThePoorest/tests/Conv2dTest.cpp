//
// Created by sidr on 10.04.23.
//

#include "gtest/gtest.h"
#include "Conv2dNaive.h"
#include "CpuBlas.h"
#include <cmath>
#include "Utils.h"

using namespace std;
using namespace nn;

TEST(Conv2d, bruh) {

    auto blas = CpuBlas();
    const vector<double> a{1, 1, 1, 2, 3,
                           1, 1, 1, 2, 3,
                           1, 1, 1, 2, 3,
                           2, 2, 2, 2, 3,
                           3, 3, 3, 3, 3,
                           4, 4, 4, 4, 4,

                           1, 1, 5, 2, 3,
                           1, 1, 6, 2, 3,
                           1, 1, 7, 2, 3,
                           2, 2, 8, 2, 3,
                           3, 3, 9, 3, 3,
                           4, 4, 0, 4, 4};

    const vector<double> w{1, 0, -1,
                           2, 0, -2,
                           1, 0, -1,
                           1, 3, -1,
                           2, 44, -2,
                           1, 2, -1,

                           3, 4, 5,
                           6, 7, 8,
                           9, 10, 11,
                           11, 23, 42,
                           2, 2, 2,
                           1, 2, 3};

    Conv2dNaive conv2D(2, 2, 3, 3, make_unique<CpuBlas>(), w);
    Tensor input(a, {1, 2, 6, 5});

    Tensor output = conv2D.forward(std::move(input));
    Tensor loss(std::vector<double>(24, 1.), {1, 2, 4, 3});
    Tensor g = conv2D.backward(loss);

    const std::vector<double> correct{4.0, 8.0, 12.0, 8.0, 4.0, 12.0, 23.0, 33.0, 21.0, 10.0, 22.0, 43.0, 63.0, 41.0,
                                      20.0,
                                      22.0, 43.0, 63.0, 41.0, 20.0, 18.0, 35.0, 51.0, 33.0, 16.0, 10.0, 20.0, 30.0,
                                      20.0,
                                      10.0, 12.0, 38.0, 79.0, 67.0, 41.0, 16.0, 88.0, 129.0, 113.0, 41.0, 18.0, 94.0,
                                      137.0, 119.0, 43.0, 18.0, 94.0, 137.0, 119.0, 43.0, 6.0,
                                      56.0, 58.0, 52.0, 2.0, 2.0, 6.0, 8.0, 6.0, 2.0};

    ASSERT_EQ(is_same_vectors(correct, g.data(), 10e-5), true);
}