//
// Created by sidr on 24.03.23.
//

#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include <vector>
#include <memory>
#include <ILayer.h>
#include "IOptimizerCreator.h"

namespace nn {

    class Sequential {
    public:
        explicit Sequential(std::vector<std::unique_ptr<ILayer>> layers,
                            std::unique_ptr<IOptimizerCreator> &&creator);

        Tensor forward(Tensor &&batch);

        Tensor backward(const Tensor &output);

        void step();

        friend std::ofstream &operator<<(std::ofstream &fout, nn::Sequential &model);

        friend std::ifstream &operator>>(std::ifstream &fin, nn::Sequential &model);

    private:
        std::vector<std::unique_ptr<IOptimizer>> optimizers_;
        std::vector<std::unique_ptr<ILayer>> layers_;
    };

    std::ofstream &operator<<(std::ofstream &fout, nn::Sequential &model);

    std::ifstream &operator>>(std::ifstream &fin, nn::Sequential &model);
}

#endif //MLP_MODEL_H
