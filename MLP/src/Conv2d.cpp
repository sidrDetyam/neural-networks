#include "Conv2d.h"
#include "Utils.h"

//
// Created by sidr on 26.03.23.
//
Conv2d::Conv2d(size_t input_channels,
               size_t output_channels,
               size_t k1, size_t k2,
               std::unique_ptr<IBlas> blas,
               std::vector<double> params) :
        input_channels_(input_channels),
        output_channels_(output_channels),
        k1_(k1),
        k2_(k2),
        blas_(std::move(blas)) {

    ASSERT(params.size() == output_channels_ * input_channels_ * k1_ * k2_);
    params_ = Tensor(std::move(params), {output_channels_, input_channels_, k1_, k2_});
    grad_ = Tensor({output_channels_, input_channels_, k1_, k2_});
}

Tensor Conv2d::forward(Tensor &&input) {

    input_copy_ = std::move(input);
    ASSERT(input_copy_.get_shape()[1] == input_channels_);
    ASSERT(input_copy_.get_shape()[2] >= k1_ && input_copy_.get_shape()[3] >= k2_);

    std::vector<size_t> input_shape = input_copy_.get_shape();
    std::vector<size_t> output_shape = input_shape;
    output_shape[1] = output_channels_;
    output_shape[2] = output_shape[2] - k1_ + 1;
    output_shape[3] = output_shape[3] - k2_ + 1;

    Tensor output(output_shape);

    for (size_t b = 0; b < output_shape[0]; ++b) {
        for (size_t out_c = 0; out_c < output_channels_; ++out_c) {
            for (size_t in_c = 0; in_c < input_channels_; ++in_c) {
                blas_->convolve(input_copy_.get_ptr({b, in_c}),
                                params_.get_ptr({out_c}),
                                output.get_ptr({b, out_c}),
                                (int)input_shape[2], (int)input_shape[3],
                                (int)k1_, (int)k2_,
                                1.);
            }
        }
    }

    return output;
}

Tensor Conv2d::backward(const Tensor &output) {
    return Tensor();
}

std::vector<double> &Conv2d::getParametersGradient() {
    return grad_.data();
}

std::vector<double> &Conv2d::getParameters() {
    return params_.data();
}
