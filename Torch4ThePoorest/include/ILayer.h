//
// Created by sidr on 18.03.23.
//

#ifndef MLP_ILAYER_H
#define MLP_ILAYER_H

#include "Tensor.h"
#include <vector>

class ILayer {
public:

    virtual Tensor forward(Tensor&& input) = 0;

    virtual Tensor backward(const Tensor& output) = 0;

    virtual std::vector<double>& getParametersGradient() = 0;

    virtual std::vector<double>& getParameters() = 0;

    virtual ~ILayer() = default;
};

#endif //MLP_ILAYER_H
