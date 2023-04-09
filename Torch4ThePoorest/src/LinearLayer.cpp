//
// Created by sidr on 18.03.23.
//

#include <utility>

#include "Tensor.h"
#include "LinearLayer.h"
#include "Utils.h"


LinearLayer::LinearLayer(size_t input_size, size_t output_size,
                         std::vector<double> weights, std::vector<double> bias,
                         std::unique_ptr<IBlas> &&blas) :
        input_size_(input_size),
        output_size_(output_size),
        blas_(std::move(blas)),
        parameters_((input_size + 1) * output_size),
        grad_((input_size + 1) * output_size){

    ASSERT_RE(input_size >= 0 && output_size >= 0 && input_size * output_size == weights.size() &&
           output_size == bias.size());

    const auto delimiter =
            parameters_.begin() + static_cast<std::iter_difference_t<std::vector<double>>>(input_size_ * output_size_);
    std::copy(weights.begin(), weights.end(), parameters_.begin());
    std::copy(bias.begin(), bias.end(), delimiter);
}

Tensor LinearLayer::forward(Tensor &&input) {

    ASSERT_RE(input.getFeatureSize() == input_size_);
    Tensor output({input.getBsize(), output_size_});

    for (size_t i = 0; i < output.getBsize(); ++i) {
        memcpy(output[i], getBPart(), output_size_ * sizeof(double));
    }

    blas_->dgemm(input[0], parameters_.data(), false, true, output[0], (int) input.getBsize(),
                 (int) output_size_, (int) input_size_, 1.);

    input_ = std::move(input);
    return output;
}

Tensor LinearLayer::backward(const Tensor &grad_output) {

    ASSERT_RE(grad_output.getBsize() == input_.getBsize() && grad_output.getFeatureSize() == output_size_);
    Tensor grad_in(input_.get_shape());

    blas_->dgemm(grad_output[0], parameters_.data(), false, false, grad_in[0],
                 (int) grad_output.getBsize(), (int) input_size_, (int) output_size_, 0);

    blas_->dgemm(grad_output[0], input_[0], true, false, grad_.data(),
                 (int) output_size_, (int) input_size_, (int) grad_output.getBsize(), 0);

    blas_->col_sum(grad_output[0], getGradBPart(), (int) grad_output.getBsize(), (int) output_size_, 0);

    blas_->scale(grad_.data(), (int) grad_.size(), 1 / (double) grad_output.getBsize());

    return grad_in;
}

std::vector<double> &LinearLayer::getParametersGradient() {
    return grad_;
}

double *LinearLayer::getBPart() {
    return parameters_.data() + input_size_ * output_size_;
}

double *LinearLayer::getGradBPart() {
    return grad_.data() + input_size_ * output_size_;
}

std::vector<double> &LinearLayer::getParameters() {
    return parameters_;
}
