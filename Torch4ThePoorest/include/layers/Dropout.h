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

        std::vector<double> &getParameters() override;

        Tensor forward(Tensor &&input) override;

        Tensor backward(const Tensor &output) override;

        std::vector<double> &getParametersGradient() override;

    private:
        double m_dropoutProbability;
        Tensor m_input;
        std::vector<bool> m_mask;
        std::vector<double> m_emptyGradient;
        std::mt19937 m_rng;
    };
}


#endif //TORCH4THEPOOREST_DROPOUT_H
