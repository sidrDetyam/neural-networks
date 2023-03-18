//
// Created by sidr on 18.03.23.
//

#include "../include/Batch.h"
#include "../include/LinearLayer.h"
#include <cassert>

const Batch& LinearLayer::forward(const Batch &input) {

    Batch out(input.getBsize(), output_size_);
    for(size_t i=0; i<out.getBsize(); ++i){
        memcpy(out[i], bias_[0], output_size_ * sizeof(double));
    }

    blas_->dgemm(input[0], weights_[0], false, true, out[0],(int)input.getBsize(), (int)output_size_, (int)input_size_, 1.);

    input_ = input;
    output_ = std::move(out);
    return output_;
}

Batch LinearLayer::backward(const Batch &grad_output) {

    Batch grad_in(grad_output.getBsize(), input_size_);
    blas_->dgemm(grad_output[0], weights_[0], false, false, grad_in[0],
                 (int)grad_output.getBsize(), (int)input_size_, (int)output_size_, 0);
    blas_->dgemm(grad_output[0], input_[0], true, false, w_grad_[0],
                 (int)output_size_, (int)input_size_, (int)grad_output.getBsize(), 0);
    blas_->col_sum(grad_output[0], b_grad_[0], (int)grad_output.getBsize(), (int)output_size_, 0);

    return grad_in;
}

LinearLayer::LinearLayer(size_t input_size, size_t output_size, std::unique_ptr<IBlas> &&blas) :
        input_size_(input_size),
        output_size_(output_size),
        blas_(std::move(blas)),
        weights_(output_size, input_size),
        bias_(1, output_size),
        w_grad_(output_size, input_size),
        b_grad_(1, output_size),
        input_(0, 0),
        output_(0, 0){

    memset(w_grad_[0], 0, output_size * input_size * sizeof(double));
    memset(b_grad_[0], 0, output_size * sizeof(double));

    std::vector<std::vector<double>> w({{1., 2., 3.}, {4., 5, 6}});

    for(int i=0; i<2; ++i){
        for(int j=0; j<3; ++j){
            weights_[i][j] = w[i][j];
        }
    }

    bias_[0][0] = 11;
    bias_[0][1] = 22;
}
