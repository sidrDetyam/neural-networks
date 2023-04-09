//
// Created by sidr on 19.03.23.
//

#include "gtest/gtest.h"
#include "../include/CrossEntropyLoss.h"
#include <cmath>

using namespace std;

TEST(Foo, Bar){
    CrossEntropyLoss loss;
    auto r = loss.apply(Tensor{{1, 2, 3., 10., 10., 5.}, {2, 3}}, {2, 0});

    ASSERT_LE(std::abs(r.first - 1.1041), 0.01);
}
