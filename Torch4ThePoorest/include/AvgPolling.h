//
// Created by sidr on 09.04.23.
//

#ifndef TORCH4THEPOOREST_AVGPOLLING_H
#define TORCH4THEPOOREST_AVGPOLLING_H

#include "ILayer.h"
#include <vector>

class AvgPolling: public ILayer{
public:
    explicit AvgPolling(size_t k1,
                        size_t k2,
                        size_t channels);

    Tensor forward(Tensor &&input) override;

    Tensor backward(const Tensor &output) override;

    std::vector<double> &getParametersGradient() override;

    std::vector<double> &getParameters() override;

private:
    std::vector<double> empty_;
    Tensor input_;
    size_t k1_;
    size_t k2_;
    size_t channels_;
};

#endif //TORCH4THEPOOREST_AVGPOLLING_H
