//
// Created by sidr on 25.03.23.
//
#include "Utils.h"
#include <random>

void random_vector_gauss(std::vector<double> &v, double mean, double dev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, dev);

    for(double & i : v){
        i = dist(gen);
    }
}

std::vector<double> random_vector_gauss(size_t n, double mean, double dev) {
    std::vector<double> res(n);
    random_vector_gauss(res, mean, dev);
    return res;
}
