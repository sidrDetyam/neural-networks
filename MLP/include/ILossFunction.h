//
// Created by sidr on 19.03.23.
//

#ifndef MLP_ILOSSFUNCTION_H
#define MLP_ILOSSFUNCTION_H

#include "Batch.h"
#include <vector>

class ILostFunction{
public:
    virtual std::pair<double, Batch> apply(const Batch& batch, const std::vector<int>& one_hot) = 0;

    virtual ~ILostFunction() = default;
};

#endif //MLP_ILOSSFUNCTION_H
