//
// Created by sidr on 21.05.23.
//

#ifndef TORCH4THEPOOREST_MAELOSS_H
#define TORCH4THEPOOREST_MAELOSS_H

#include "IRegressionLossFunction.h"

namespace nn{

    class MAELoss: public IRegressionLossFunction{
    public:
        std::pair<double, Tensor> apply(const Tensor &model_output, const Tensor &correct_output) override;
    };

}

#endif //TORCH4THEPOOREST_MAELOSS_H
