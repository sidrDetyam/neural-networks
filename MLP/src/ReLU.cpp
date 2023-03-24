#include <vector>
#include "Batch.h"
#include "ReLU.h"
#include "Utils.h"

//
// Created by sidr on 24.03.23.
//
Batch ReLU::forward(Batch &&input) {

    if(!mask_.isSameShape(input)) {
        Batch m(input.getBsize(), input.getFeatureSize());
        mask_ = std::move(m);
    }

    for(size_t i=0; i<input.getBsize() * input.getFeatureSize(); ++i){
        *mask_[i] = static_cast<double>(*input[i] >= 0);
        *input[i] = std::max(*input[i], 0.);
    }

    return input;
}

Batch ReLU::backward(const Batch &output) {

    ASSERT(mask_.isSameShape(output));

    Batch input(mask_.getBsize(), mask_.getFeatureSize());
    //TODO
    for(size_t i=0; i<input.getBsize() * input.getFeatureSize(); ++i){
        *input[i] = *output[i] * *mask_[i];
    }

    return input;
}

std::vector<double> &ReLU::getParametersGradient() {
    return fiction_grad_;
}

std::vector<double> &ReLU::getParameters() {
    return fiction_grad_;
}
