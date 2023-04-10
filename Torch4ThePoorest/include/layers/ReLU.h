//
// Created by sidr on 24.03.23.
//

#ifndef MLP_RELU_H
#define MLP_RELU_H

#include "ILayer.h"
#include <vector>
#include "IBlas.h"

namespace nn {

    class ReLU : public ILayer {
    public:
        explicit ReLU(std::unique_ptr<IBlas> &&blas);

        Tensor forward(Tensor &&input) override;

        Tensor backward(const Tensor &output) override;

        std::vector<double> &getParametersGradient() override;

        std::vector<double> &getParameters() override;

    private:
        Tensor mask_;
        std::vector<double> fiction_grad_;
        std::unique_ptr<IBlas> blas_;
    };
}

#endif //MLP_RELU_H
