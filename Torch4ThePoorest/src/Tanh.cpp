//
// Created by sidr on 11.04.23.
//

#include <cmath>
#include "Tanh.h"

using namespace nn;

Tensor Tanh::forward(Tensor &&input) {
    Tensor output(input.get_shape());

    for (size_t i = 0; i < input.data().size(); i++) {
        output.data()[i] = tanh(input.data()[i]);
    }
    input_ = std::move(input);

    return output;
}

Tensor Tanh::backward(const Tensor &grad_output) {
    Tensor grad_input(grad_output.get_shape());

    for (size_t i = 0; i < grad_output.data().size(); i++) {
        double grad = grad_output.data()[i];
        double out = tanh(input_.data()[i]);
        grad_input.data()[i] = grad * (1 - out * out);
    }

    return grad_input;
}

std::vector<double> &Tanh::getParametersGradient() {
    return empty_;
}

std::vector<double> &Tanh::getParameters() {
    return empty_;
}
