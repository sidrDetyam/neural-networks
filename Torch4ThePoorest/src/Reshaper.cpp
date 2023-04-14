#include "Tensor.h"
#include "Reshaper.h"
#include "Utils.h"

//
// Created by sidr on 10.04.23.
//
nn::Tensor nn::Reshaper::forward(Tensor &&input) {
    tshape_t in_shape = input.get_shape();
    ASSERT_RE(!in_shape.empty());
    in_shape.erase(in_shape.begin());
    ASSERT_RE(in_shape == is_);

    tshape_t out_shape = os_;
    out_shape.insert(out_shape.begin(), input.get_shape()[0]);
    input.reshape({out_shape});

    return input;
}

nn::Tensor nn::Reshaper::backward(const Tensor &output) {
    Tensor input = output;
    tshape_t out_shape = output.get_shape();
    ASSERT_RE(!out_shape.empty());
    out_shape.erase(out_shape.begin());
    ASSERT_RE(out_shape == os_);

    tshape_t in_shape = is_;
    in_shape.insert(in_shape.begin(), output.get_shape()[0]);
    input.reshape({in_shape});

    return input;
}

std::vector<double> &nn::Reshaper::getParametersGradient() {
    return empty_;
}

std::vector<double> &nn::Reshaper::getParameters() {
    return empty_;
}

nn::Reshaper::Reshaper(nn::tshape_t is, nn::tshape_t os): is_(std::move(is)), os_(std::move(os)) {
    ASSERT_RE(is_same_cnt(is_, os_));
}
