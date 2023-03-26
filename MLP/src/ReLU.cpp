#include <vector>
#include "Tensor.h"
#include "ReLU.h"
#include "Utils.h"

//
// Created by sidr on 24.03.23.
//
Tensor ReLU::forward(Tensor &&input) {

    if(!mask_.isSameShape(input)){
        mask_ = input;
    }

    /// TODO
    for(size_t i=0; i<input.getBsize(); ++i){
        for(size_t j=0; j<input.getFeatureSize(); ++j){
            mask_[i][j] = static_cast<double>(input[i][j] >= 0);
            input[i][j] = std::max(input[i][j], 0.);
        }
    }

    return input;
}

Tensor ReLU::backward(const Tensor &output) {

    ASSERT(mask_.isSameShape(output));

    Tensor input(mask_.get_shape());

    for(size_t i=0; i<input.getBsize(); ++i){
        for(size_t j=0; j<input.getFeatureSize(); ++j){
            input[i][j] = output[i][j] * mask_[i][j];
        }
    }

    //blas_->daxpby(output.getBsize() * output.getFeatureSize(), out)

    return input;
}

std::vector<double> &ReLU::getParametersGradient() {
    return fiction_grad_;
}

std::vector<double> &ReLU::getParameters() {
    return fiction_grad_;
}

//ReLU::ReLU(std::unique_ptr<IBlas> &&blas): blas_(std::move(blas)) {
//
//}
