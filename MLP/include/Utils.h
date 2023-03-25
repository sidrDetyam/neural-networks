//
// Created by sidr on 19.03.23.
//

#ifndef MLP_UTILS_H
#define MLP_UTILS_H

#include <iostream>
#include <vector>

#define ASSERT(cond__) \
do{\
    if(!(cond__)){     \
        std::cerr << #cond__ << std::endl;     \
        std::abort(); \
    } \
}while(0)


void random_vector_gauss(std::vector<double>& v, double mean, double dev);

std::vector<double> random_vector_gauss(size_t n, double mean, double dev);

#endif //MLP_UTILS_H
