//
// Created by sidr on 12.04.23.
//
#include "Conv2d.h"
#include "Utils.h"

using namespace nn;

nn::Tensor nn::Conv2d::forward(nn::Tensor &&input) {
    return Tensor();
}

nn::Tensor nn::Conv2d::backward(const nn::Tensor &output) {
    return Tensor();
}

std::vector<double> &nn::Conv2d::getParametersGradient() {
    return grad_.data();
}

std::vector<double> &nn::Conv2d::getParameters() {
    return params_.data();
}

Conv2d::Conv2d(const size_t input_channels,
               const size_t output_channels,
               const size_t kernel_,
               std::unique_ptr<IBlas> blas,
               std::vector<double> params) :
        input_channels_(input_channels),
        output_channels_(output_channels),
        kernel_(kernel_),
        blas_(std::move(blas)) {

    ASSERT_RE(params.size() == output_channels_ * input_channels_ * kernel_ * kernel_);
    params_ = Tensor(std::move(params), {output_channels_, input_channels_, kernel_, kernel_});
    grad_ = Tensor({output_channels_, input_channels_, kernel_, kernel_});
}

void Conv2d::img2col(const double *const original,
                     const size_t h, const size_t w,
                     const size_t kernel,
                     double *const res) {
    ASSERT_RE(h >= kernel && w >= kernel);

    //bruh
    for (size_t kh = 0; kh < kernel; ++kh) {
        for (size_t kw = 0; kw < kernel; ++kw) {
            for (size_t i = 0; i < h - kernel + 1; ++i) {
                for (size_t j = 0; j < w - kernel + 1; ++j) {
                    res[(kh * kernel + kw) * ((h - kernel + 1) * (w - kernel + 1)) +
                        (i * (w - kernel + 1) + j)] = original[(i + kh) * (w) + (j + kw)];
                }
            }
        }
    }
}
