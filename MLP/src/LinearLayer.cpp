//
// Created by sidr on 18.03.23.
//

#include "../include/Batch.h"
#include "../include/LinearLayer.h"

Batch LinearLayer::forward(const Batch &input) {
    return {0, 0};
}

Batch LinearLayer::backward(const Batch &output) {
    return {0, 0};
}

LinearLayer::LinearLayer(size_t input_size, size_t output_size, std::unique_ptr<IBlas> &&blas):
    input_size_(input_size),
    output_size_(output_size),
    blas_(std::move(blas)){

}
