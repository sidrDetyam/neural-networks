//
// Created by sidr on 21.05.23.
//
#include "RnnCell.h"
#include "Utils.h"

nn::RnnCell::RnnCell(const size_t input_size, const size_t hidden_size,
                     std::unique_ptr<IActivation> &&activation,
                     std::function<IBlas*()> &&blas_factory):
                     input_size_(input_size),
                     hidden_size_(hidden_size),
                     activation_(std::move(activation)),
                     dense_(input_size + hidden_size, hidden_size, std::unique_ptr<IBlas>(blas_factory())),
                     blas_factory_(std::move(blas_factory)){
    ASSERT_RE(input_size_ > 0 && hidden_size_ > 0 && activation_);
}

void nn::RnnCell::forward(nn::Tensor &&input_tensor, nn::Tensor &&hidden_tensor,
                          const double * const params) {

    const auto& its = input_tensor.get_shape();

    ASSERT_RE(its.size() == 2 && its[1] == input_size_);
    ASSERT_RE(hidden_tensor.get_shape().size() == 2 && hidden_tensor.get_shape()[1] == hidden_size_);

    Tensor input({its[0], input_size_+hidden_size_});
    for(size_t b=0; b<its[0]; ++b){
        std::memcpy(&input({b}), &input_tensor({b}), sizeof(double)*input_size_);
        std::memcpy(&input({b, input_size_}), &hidden_tensor({b}), sizeof(double)*hidden_size_);
    }

    auto& dense_parameters = dense_.getParameters();
    std::memcpy(dense_parameters.data(), params, dense_parameters.size() * sizeof(double));
    output_ = dense_.forward(std::move(input));
    output_ = activation_->forward(std::move(output_));
}

void nn::RnnCell::backward(const nn::Tensor &output_grad, double * const grad_ptr) {

    Tensor grad = activation_->backward(output_grad);
    auto& dense_grad = dense_.getParametersGradient();

    dense_grad.assign(dense_grad.size(), 0.);
    grad = dense_.backward(grad);
    auto blas = std::unique_ptr<IBlas>(blas_factory_());
    blas->daxpby((int) dense_grad.size(), dense_grad.data(), 1., grad_ptr, 1.);

    const size_t batch_size = grad.get_shape()[0];
    Tensor input_grad({batch_size, input_size_});
    Tensor hidden_grad({batch_size, hidden_size_});

    for(size_t b=0; b<batch_size; ++b){
        std::memcpy(&input_grad({b}), &grad({b}), sizeof(double)*input_size_);
        std::memcpy(&hidden_grad({b}), &grad({b, input_size_}), sizeof(double)*hidden_size_);
    }
    input_grad_ = std::move(input_grad);
    hidden_grad_ = std::move(hidden_grad);
}

nn::Tensor &nn::RnnCell::get_input_grad() {
    return input_grad_;
}

nn::Tensor &nn::RnnCell::get_hidden_grad() {
    return hidden_grad_;
}
