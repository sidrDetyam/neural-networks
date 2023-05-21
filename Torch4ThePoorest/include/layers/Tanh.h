//
// Created by sidr on 11.04.23.
//

#ifndef TORCH4THEPOOREST_TANH_H
#define TORCH4THEPOOREST_TANH_H

#include "IActivation.h"

namespace nn{

    class Tanh : public ILayer {
    public:
        Tensor forward(Tensor&& input) override;

        Tensor backward(const Tensor& grad_output) override;

    private:
        Tensor input_;
    };
}

#endif //TORCH4THEPOOREST_TANH_H
