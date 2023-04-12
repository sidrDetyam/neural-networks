//
// Created by sidr on 12.04.23.
//
#include "Conv2d.h"
#include "Utils.h"

using namespace nn;

nn::Tensor nn::Conv2d::forward(nn::Tensor &&input) {

    input_copy_ = std::move(input);
    ASSERT_RE(input_copy_.get_shape()[1] == input_channels_);
    ASSERT_RE(input_copy_.get_shape()[2] >= kernel_ && input_copy_.get_shape()[3] >= kernel_);

    const std::vector<size_t> &input_shape = input_copy_.get_shape();
    const std::vector<size_t> output_shape = get_output_shape(input_shape);
    Tensor output(output_shape);

    buff_.resize({input_channels_, kernel_ * kernel_, output_shape[2] * output_shape[3]});

    for (size_t b = 0; b < input_shape[0]; ++b) {
        for (size_t c_in = 0; c_in < input_channels_; ++c_in) {
            img2col(&input_copy_({b, c_in}),
                    input_shape[2], input_shape[3], kernel_,
                    &buff_({c_in}));
        }

        for (size_t c_out = 0; c_out < output_channels_; ++c_out) {
            for (size_t c_in = 0; c_in < input_channels_; ++c_in) {
                blas_->dgemm_full(ROW_ORDER, NO_TRANS, NO_TRANS,
                                  (int) 1, (int) (output_shape[2] * output_shape[3]),
                                  (int) (kernel_ * kernel_),
                                  1., &params_({c_out, c_in}), (int) (kernel_ * kernel_),
                                  &buff_({c_in}), (int) (output_shape[2] * output_shape[3]), 1.,
                                  &output({b, c_out}), (int) (output_shape[2] * output_shape[3]));
            }
        }
    }

    return output;
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

std::vector<size_t> Conv2d::get_output_shape(const std::vector<size_t> &input_shape) const {
    ASSERT_RE(input_shape.size() == 4);
    ASSERT_RE(input_copy_.get_shape()[1] == input_channels_);
    ASSERT_RE(input_copy_.get_shape()[2] >= kernel_ && input_copy_.get_shape()[3] >= kernel_);

    std::vector<size_t> output_shape = input_shape;
    output_shape[1] = output_channels_;
    output_shape[2] = output_shape[2] - kernel_ + 1;
    output_shape[3] = output_shape[3] - kernel_ + 1;
    return output_shape;
}
