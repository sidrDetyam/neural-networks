//
// Created by sidr on 11.04.23.
//

#ifndef TORCH4THEPOOREST_DROPOUT_H
#define TORCH4THEPOOREST_DROPOUT_H

#include <random>
#include <vector>
#include "ILayer.h"
#include "Utils.h"

namespace nn{

    class DropoutLayer : public ILayer {
    public:
        [[maybe_unused]] explicit DropoutLayer(double dropoutProbability);

        Tensor forward(Tensor &&input) override;

        Tensor backward(const Tensor &output) override;

    private:
        double m_dropoutProbability;
        Tensor m_input;
        std::vector<bool> m_mask;
        std::mt19937 m_rng;
    };
}


#endif //TORCH4THEPOOREST_DROPOUT_H
