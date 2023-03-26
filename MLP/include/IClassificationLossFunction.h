//
// Created by sidr on 19.03.23.
//

#ifndef MLP_ICLASSIFICATIONLOSSFUNCTION_H
#define MLP_ICLASSIFICATIONLOSSFUNCTION_H

#include "Batch.h"
#include <vector>

class IClassificationLostFunction{
public:
    virtual std::pair<double, Batch> apply(const Batch& batch, const std::vector<int>& one_hot) = 0;

    virtual ~IClassificationLostFunction() = default;
};

#endif //MLP_ICLASSIFICATIONLOSSFUNCTION_H
