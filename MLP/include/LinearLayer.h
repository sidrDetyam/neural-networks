//
// Created by sidr on 18.03.23.
//

#ifndef MLP_LINEARLAYER_H
#define MLP_LINEARLAYER_H

#include "ILayer.h"
#include "IBlas.h"
#include <vector>

class LinearLayer : public ILayer {
public:
    explicit LinearLayer(size_t input_size, size_t output_size,
                         std::vector<double> weights,
                         std::vector<double> bias,
                         std::unique_ptr<IBlas> &&blas);

    Batch forward(Batch &&input) override;

    Batch backward(const Batch &output) override;

    std::vector<double> &getParametersGradient() override;

    std::vector<double> &getParameters() override;

private:
    double *getBPart();

    double *getGradBPart();

    std::unique_ptr<IBlas> blas_;
    const size_t input_size_;
    const size_t output_size_;

    std::vector<double> parameters_;
    std::vector<double> grad_;
    Batch input_;
};

#endif //MLP_LINEARLAYER_H
