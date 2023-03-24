//
// Created by sidr on 24.03.23.
//

#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include "Optimizer.h"
#include <vector>
#include <memory>
#include <ILayer.h>

class Model{
public:
    explicit Model(std::vector<std::unique_ptr<ILayer>> layers);

    

private:
    std::vector<std::unique_ptr<ILayer>> layers_;
    std::vector<Optimizer> optimizers_;
};

#endif //MLP_MODEL_H
