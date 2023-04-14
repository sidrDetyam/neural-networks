//
// Created by sidr on 19.03.23.
//

#ifndef MLP_ICLASSIFICATIONLOSSFUNCTION_H
#define MLP_ICLASSIFICATIONLOSSFUNCTION_H

#include "Tensor.h"
#include <vector>

namespace nn {

    class IClassificationLostFunction {
    public:
        virtual std::pair<double, Tensor> apply(const Tensor &batch,
                                                const std::vector<int> &one_hot) = 0;

        virtual ~IClassificationLostFunction() = default;
    };
}

#endif //MLP_ICLASSIFICATIONLOSSFUNCTION_H
