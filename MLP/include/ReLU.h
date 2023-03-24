//
// Created by sidr on 24.03.23.
//

#ifndef MLP_RELU_H
#define MLP_RELU_H

#include "ILayer.h"
#include <vector>

class ReLU : public ILayer{
public:
    Batch forward(Batch &&input) override;

    Batch backward(const Batch &output) override;

    std::vector<double> &getParametersGradient() override;

    std::vector<double> &getParameters() override;

private:
    Batch mask_;
    std::vector<double> fiction_grad_;
};


#endif //MLP_RELU_H
