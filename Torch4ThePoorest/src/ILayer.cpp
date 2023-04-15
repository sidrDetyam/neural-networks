//
// Created by sidr on 15.04.23.
//

#include "ILayer.h"

std::vector<double> &nn::ILayer::getParametersGradient() {
    return grad_;
}

std::vector<double> &nn::ILayer::getParameters() {
    return params_;
}
