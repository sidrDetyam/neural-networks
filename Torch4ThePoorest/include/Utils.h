//
// Created by sidr on 19.03.23.
//

#ifndef MLP_UTILS_H
#define MLP_UTILS_H

#include <iostream>
#include <vector>
#include <numeric>

//        std::cerr << #cond__ << std::endl;     \
//        std::abort(); \

#define ASSERT_RE(cond__) \
do{\
    if(!(cond__)){     \
        throw std::runtime_error(#cond__); \
    } \
}while(0)

[[maybe_unused]] bool is_same_vectors(const std::vector<double> &a, const std::vector<double> &b, double eps);

bool is_same_cnt(const std::vector<size_t> &a, const std::vector<size_t> &b);

#endif //MLP_UTILS_H
