//
// Created by sidr on 10.04.23.
//

#include "gtest/gtest.h"
#include "Conv2dNaive.h"
#include "Conv2d.h"
#include "CpuBlas.h"
#include <cmath>
#include "Utils.h"

using namespace std;
using namespace nn;

TEST(Conv2d, naive) {
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

TEST(Conv2d, fast_forward){
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
                           4, 4, 0, 4, 4
    };

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
                           1, 2, 3
    };

    Conv2dNaive conv2DNaive(2, 2, 3, 3, make_unique<CpuBlas>(), w);
    Conv2d conv2D(2, 2, 3, make_unique<CpuBlas>(), w);

    Tensor input(a, {1, 2, 6, 5});

    Tensor output = conv2DNaive.forward(Tensor(input));
    Tensor output2 = conv2D.forward(Tensor(input));

    ASSERT_EQ(is_same_vectors(output.data(), output2.data(), 1e-5), true);
}

TEST(Conv2d, fast_backward){
    const size_t in = 16;
    const size_t out = 120;
    const size_t w = 5;
    const size_t h = 5;
    const size_t k = 4;
    const size_t b = 10000;
    const size_t rounds = 1;
    const auto weights = random_vector_gauss(in * out * k * k, 0, 3);

    Conv2dNaive conv2DNaive(in, out, k, k, make_unique<CpuBlas>(), weights);
    Conv2d conv2D(in, out, k, make_unique<CpuBlas>(), weights);

    Tensor input(xavier_init(b*in*w*h), {b, in, h, w});
    Tensor l(random_vector_gauss(b*out*(w-k+1)*(h - k + 1), 0, 3), {b, out, h-k+1, w-k+1});

    Tensor output;
    Tensor output2;

    Tensor g1, g2;
    using namespace chrono;

    auto n1 = high_resolution_clock::now();
    for(int i=0; i<rounds; ++i) {
        output = conv2DNaive.forward(Tensor(input));
        g1 = conv2DNaive.backward(l);
    }
    auto n2 = high_resolution_clock::now();

    auto p1 = high_resolution_clock::now();
    for(int i=0; i<rounds; ++i) {
        output2 = conv2D.forward(Tensor(input));
        g2 = conv2D.backward(l);
    }
    auto p2 = high_resolution_clock::now();

    auto f1 = high_resolution_clock::now();
    for(int i=0; i<rounds; ++i) {
        output2 = conv2D.forward(Tensor(input));
        g2 = conv2D.backward(l);
    }
    auto f2 = high_resolution_clock::now();

    auto n = duration_cast<microseconds>(n2 - n1).count();
    auto f = duration_cast<microseconds>(f2 - f1).count();
    auto  p = duration_cast<microseconds>(p2-p1).count();

    cout << n << " " << f << " " << p <<  endl;

    ASSERT_EQ(is_same_vectors(output.data(), output2.data(), 1e-5), true);
    ASSERT_EQ(is_same_vectors(conv2D.getParametersGradient(), conv2DNaive.getParametersGradient(), 1e-5), true);
    ASSERT_EQ(is_same_vectors(g1.data(), g2.data(), 1e-3), true);
}

//TEST(Conv2d, speed_forward){
//    const size_t inc = 6;
//    const size_t outc = 16;
//    const size_t k = 5;
//    const size_t b = 12;
//    const size_t h = 12;
//    const int rounds = 10;
//
//    vector<double> w1(inc * outc * k * k);
//
//    Conv2dNaive conv2DNaive(inc, outc, k, k, make_unique<CpuBlas>(), w1);
//    Conv2d conv2D(inc, outc, k, make_unique<CpuBlas>(), w1);
//
//    Tensor input(vector<double>(inc*b*h*h, 1.), {b, inc, h, h});
//
//    using namespace chrono;
//
//    Tensor output, output2;
//
//    auto n1 = high_resolution_clock::now();
//    for(int i=0; i<rounds; ++i) {
//        output = conv2DNaive.forward(Tensor(input));
//    }
//    auto n2 = high_resolution_clock::now();
//
//    auto f1 = high_resolution_clock::now();
//    for(int i=0; i<rounds; ++i) {
//        output2 = conv2D.forward(Tensor(input));
//        //conv2D.backward()
//    }
//    auto f2 = high_resolution_clock::now();
//
//    auto n = duration_cast<microseconds>(n2 - n1).count();
//    auto f = duration_cast<microseconds>(f2 - f1).count();
//
//    cout << n << " " << f << endl;
//
//    ASSERT_EQ(true, is_same_vectors(output.data(), output2.data(), 1e-5));
//    ASSERT_GE(n, f);
//}

TEST(Conv2d, padding){
    Tensor a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 3});
    Tensor b(vector<double>(36, 1), {6, 6});
    Conv2d::add_padding(&a({0}), &b({0}), 4, 3, 2, 0, 1, 2);

    cout << b << endl;
}