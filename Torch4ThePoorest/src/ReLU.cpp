#include <vector>
#include "Tensor.h"
#include "ReLU.h"
#include "Utils.h"

//
// Created by sidr on 24.03.23.
//
using namespace nn;

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

    ASSERT_RE(mask_.isSameShape(output));

    Tensor input(mask_.get_shape());
    blas_->element_wise_mult((int)(input.getBsize() * input.getFeatureSize()),
                             output[0], mask_[0], input[0]);

    return input;
}

ReLU::ReLU(std::unique_ptr<IBlas> &&blas): blas_(std::move(blas)) {

}
