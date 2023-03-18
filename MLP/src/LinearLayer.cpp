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

    blas_->dgemm(input[0], weights_[0], false, true, out[0],
                 (int)input.getBsize(), (int)output_size_, (int)input_size_, 1.);

    input_ = std::move(out);
    retur;
}

Batch LinearLayer::backward(const Batch &output) {
    return {0, 0};
}

LinearLayer::LinearLayer(size_t input_size, size_t output_size, std::unique_ptr<IBlas> &&blas) :
        input_size_(input_size),
        output_size_(output_size),
        blas_(std::move(blas)),
        weights_(output_size, input_size),
        bias_(1, output_size),
        w_grad_(output_size, input_size),
        b_grad_(1, output_size),
        input_(0, 0){

    memset(w_grad_[0], 0, output_size * input_size * sizeof(double));
    memset(b_grad_[0], 0, output_size * sizeof(double));
}
