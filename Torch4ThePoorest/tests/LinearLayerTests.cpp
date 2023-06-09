//
// Created by sidr on 19.03.23.
//

#include "gtest/gtest.h"
#include "Linear.h"
#include "CpuBlas.h"

using namespace nn;

TEST(LL, forward){
    Linear ll(3, 2, {1., 2., 3., 4., 5, 6}, {11, 22}, CpuBlas::of());
    Tensor input(Tensor{{1, 2, 3}, {1, 3}});
    auto output = ll.forward(std::move(input));
    ASSERT_EQ(output.getBsize(), 1) << "Incorrect batch size";
    ASSERT_EQ(output.getFeatureSize(), 2) << "Incorrect features size";

    ASSERT_EQ(output[0][0], 25);
    ASSERT_EQ(output[0][1], 54);
}


TEST(LL, backward){
    Linear ll(3, 2, {1., 2., 3., 4., 5, 6}, {11, 22}, CpuBlas::of());
    Tensor input({1, 2, 3, 1, 2, 3}, {2, 3});
    ll.forward(std::move(input));

    //auto grad_b0 = ll.backward({{90, 168}});
    auto grad_b = ll.backward(Tensor{{90, 168, 90, 168}, {2, 2}});
    auto grad_p = ll.getParametersGradient();

    std::vector<double> expected_grad_p{90., 180., 270., 168., 336., 504., 90, 168};
    std::vector<double> expected_grad_b{762, 1020, 1278};

    for(size_t i=0; i<8; ++i){
        ASSERT_EQ(expected_grad_p[i], grad_p[i]);
    }

    for(size_t i=0; i<3; ++i){
        ASSERT_EQ(expected_grad_b[i], grad_b[0][i]);
    }
}