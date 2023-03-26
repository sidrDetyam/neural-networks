#include <vector>
#include "Batch.h"
#include "ReLU.h"
#include "Utils.h"

//
// Created by sidr on 24.03.23.
//
Batch ReLU::forward(Batch &&input) {

    if(!mask_.isSameBandFsize(input)) {
        Batch m(input.getBsize(), {input.getFeatureSize()});
        mask_ = std::move(m);
    }

    for(size_t i=0; i<input.getBsize(); ++i){
        for(size_t j=0; j<input.getFeatureSize(); ++j){
            mask_[i][j] = static_cast<double>(input[i][j] >= 0);
            input[i][j] = std::max(input[i][j], 0.);
        }
    }

    return input;
}

Batch ReLU::backward(const Batch &output) {

    ASSERT(mask_.isSameBandFsize(output));

    Batch input(mask_.getBsize(), {mask_.getFeatureSize()});

    for(size_t i=0; i<input.getBsize(); ++i){
        for(size_t j=0; j<input.getFeatureSize(); ++j){
            input[i][j] = output[i][j] * mask_[i][j];
        }
    }

    return input;
}

std::vector<double> &ReLU::getParametersGradient() {
    return fiction_grad_;
}

std::vector<double> &ReLU::getParameters() {
    return fiction_grad_;
}
