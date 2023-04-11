//
// Created by sidr on 11.04.23.
//

#ifndef TORCH4THEPOOREST_TANH_H
#define TORCH4THEPOOREST_TANH_H

#include "ILayer.h"

namespace nn{

    class Tanh : public ILayer {
    public:
        Tensor forward(Tensor&& input) override;

        Tensor backward(const Tensor& grad_output) override;

        std::vector<double>& getParametersGradient() override;

        std::vector<double>& getParameters() override;

    private:
        Tensor input_;
        std::vector<double> empty_;
    };
}

#endif //TORCH4THEPOOREST_TANH_H
