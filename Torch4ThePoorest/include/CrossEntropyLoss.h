//
// Created by sidr on 19.03.23.
//

#ifndef MLP_CROSSENTROPYLOSS_H
#define MLP_CROSSENTROPYLOSS_H

#include "IClassificationLossFunction.h"

namespace nn {

    class CrossEntropyLoss : public IClassificationLostFunction {
    public:
        std::pair<double, Tensor> apply(const Tensor &batch, const std::vector<int> &one_hot) override;

        static Tensor softMax(const Tensor &batch);
    };
}

#endif //MLP_CROSSENTROPYLOSS_H
