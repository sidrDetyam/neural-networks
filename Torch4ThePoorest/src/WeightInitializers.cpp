//
// Created by sidr on 15.04.23.
//

#include "WeightInitializers.h"
#include <random>

void nn::random_vector_gauss(std::vector<double> &v, double mean, double dev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, dev);

    for (double &i: v) {
        i = dist(gen);
    }
}

[[maybe_unused]] std::vector<double> nn::random_vector_gauss(size_t n, double mean, double dev) {
    std::vector<double> res(n);
    random_vector_gauss(res, mean, dev);
    return res;
}

[[maybe_unused]] std::vector<double> nn::xavier_init(size_t n) {
    return nn::random_vector_gauss(n, 0, sqrt(2. / (double)n));
}

