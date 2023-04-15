//
// Created by sidr on 15.04.23.
//

#ifndef TORCH4THEPOOREST_WEIGHTINITIALIZERS_H
#define TORCH4THEPOOREST_WEIGHTINITIALIZERS_H

#include <vector>

namespace nn{
    void random_vector_gauss(std::vector<double> &v, double mean, double dev);

    [[maybe_unused]] std::vector<double> random_vector_gauss(size_t n, double mean, double dev);

    [[maybe_unused]] std::vector<double> xavier_init(size_t n);
}

#endif //TORCH4THEPOOREST_WEIGHTINITIALIZERS_H
