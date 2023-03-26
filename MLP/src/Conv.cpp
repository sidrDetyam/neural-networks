#include "Conv.h"
#include "Utils.h"

//
// Created by sidr on 26.03.23.
//
Batch Conv::forward(Batch &&input) {

    //input_copy_ = std::move(input);

    return Batch();
}

Batch Conv::backward(const Batch &output) {
    return Batch();
}

std::vector<double> &Conv::getParametersGradient() {
    return grad_;
}

std::vector<double> &Conv::getParameters() {
    return params_;
}

Conv::Conv(std::pair<size_t, size_t> shape,
           std::unique_ptr<IBlas> blas,
           std::vector<double> params):
           shape_(std::move(shape)),
           blas_(std::move(blas)),
           params_(std::move(params)){

    ASSERT(params_.size() == shape_.first * shape_.second);
    grad_.assign(shape_.first * shape_.second, 0.);
}
