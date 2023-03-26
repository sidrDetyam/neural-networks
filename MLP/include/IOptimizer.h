//
// Created by sidr on 26.03.23.
//

#ifndef MLP_IOPTIMIZER_H
#define MLP_IOPTIMIZER_H

class IOptimizer{
public:
    virtual void step() = 0;

    virtual ~IOptimizer() = default;
};

#endif //MLP_IOPTIMIZER_H
