#include "Conv2d.h"
#include "Utils.h"

//
// Created by sidr on 26.03.23.
//
Tensor Conv2d::forward(Tensor &&input) {

    //input_copy_ = std::move(input);

    return Tensor();
}

Tensor Conv2d::backward(const Tensor &output) {
    return Tensor();
}

std::vector<double> &Conv2d::getParametersGradient() {
    return grad_;
}

std::vector<double> &Conv2d::getParameters() {
    return params_;
}

Conv2d::Conv2d(std::pair<size_t, size_t> shape,
               std::unique_ptr<IBlas> blas,
               std::vector<double> params):
           shape_(std::move(shape)),
           blas_(std::move(blas)),
           params_(std::move(params)){

    ASSERT(params_.size() == shape_.first * shape_.second);
    grad_.assign(shape_.first * shape_.second, 0.);
}
