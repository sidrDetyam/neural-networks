//
// Created by sidr on 19.03.23.
//

#ifndef MLP_CROSSENTROPYLOSS_H
#define MLP_CROSSENTROPYLOSS_H

#include "IClassificationLossFunction.h"

class CrossEntropyLoss: public IClassificationLostFunction{
public:
    std::pair<double, Batch> apply(const Batch &batch, const std::vector<int>& one_hot) override;

    static Batch softMax(const Batch &batch);
};

#endif //MLP_CROSSENTROPYLOSS_H
