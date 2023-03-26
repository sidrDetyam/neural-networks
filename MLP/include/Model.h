//
// Created by sidr on 24.03.23.
//

#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include <vector>
#include <memory>
#include <ILayer.h>
#include "IOptimizerCreator.h"


class Model{
public:
    explicit Model(std::vector<std::unique_ptr<ILayer>> layers,
                   std::unique_ptr<IOptimizerCreator>&& creator);

    Batch forward(Batch&& batch);

    void backward(const Batch& output);

    void step();

private:
    std::vector<std::unique_ptr<ILayer>> layers_;
    std::vector<std::unique_ptr<IOptimizer>> optimizers_;
};

#endif //MLP_MODEL_H
