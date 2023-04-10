//
// Created by sidr on 26.03.23.
//

#ifndef MLP_IOPTIMIZERCREATOR_H
#define MLP_IOPTIMIZERCREATOR_H

#include "IOptimizer.h"
#include "ILayer.h"

namespace nn {

    class IOptimizerCreator {
    public:
        virtual IOptimizer *create(ILayer *layer) = 0;

        virtual ~IOptimizerCreator() = default;
    };
}

#endif //MLP_IOPTIMIZERCREATOR_H
