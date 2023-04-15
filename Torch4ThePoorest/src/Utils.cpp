//
// Created by sidr on 25.03.23.
//
#include "Utils.h"

[[maybe_unused]] bool is_same_vectors(const std::vector<double> &a, const std::vector<double> &b, double eps) {
    if (a.size() != b.size()) {
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > eps) {
            return false;
        }
    }

    return true;
}

bool is_same_cnt(const std::vector<size_t> &a, const std::vector<size_t> &b) {
    return std::reduce(a.begin(), a.end(), 1ul, std::multiplies<>()) ==
           std::reduce(b.begin(), b.end(), 1ul, std::multiplies<>());
}
