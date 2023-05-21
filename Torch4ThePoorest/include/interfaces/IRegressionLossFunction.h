//
// Created by sidr on 21.05.23.
//

#ifndef TORCH4THEPOOREST_IREGRESSIONLOSSFUNCTION_H
#define TORCH4THEPOOREST_IREGRESSIONLOSSFUNCTION_H

#include "Tensor.h"

namespace nn {

    class IRegressionLossFunction {
    public:
        [[nodiscard]] virtual std::pair<double, Tensor> apply(const Tensor &model_output,
                                                              const Tensor &correct_output) = 0;

        virtual ~IRegressionLossFunction() = default;
    };
}

#endif //TORCH4THEPOOREST_IREGRESSIONLOSSFUNCTION_H
