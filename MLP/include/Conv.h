//
// Created by sidr on 26.03.23.
//

#ifndef MLP_CONV_H
#define MLP_CONV_H

#include "ILayer.h"
#include "IBlas.h"

class Conv: public ILayer{
public:
    explicit Conv(std::pair<size_t, size_t> shape,
                  std::unique_ptr<IBlas> blas,
                  std::vector<double> params);

    Tensor forward(Tensor &&input) override;

    Tensor backward(const Tensor &output) override;

    std::vector<double> &getParametersGradient() override;

    std::vector<double> &getParameters() override;

private:
    std::unique_ptr<IBlas> blas_;
    Tensor input_copy_;
    std::pair<size_t, size_t> shape_;
    std::vector<double> params_;
    std::vector<double> grad_;
};

#endif //MLP_CONV_H
