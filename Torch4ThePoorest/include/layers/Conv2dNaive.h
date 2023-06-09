//
// Created by sidr on 26.03.23.
//

#ifndef MLP_CONV2D_H
#define MLP_CONV2D_H

#include "ILayer.h"
#include "IBlas.h"

//TODO stride and padding
//TODO make it more efficient

namespace nn {

    class Conv2dNaive : public ILayer {
    public:
        explicit Conv2dNaive(size_t input_channels,
                             size_t output_channels,
                             size_t k1,
                             size_t k2,
                             std::unique_ptr<IBlas> blas,
                             std::vector<double> params);

        Tensor forward(Tensor &&input) override;

        Tensor backward(const Tensor &output) override;

        std::vector<double> &getParametersGradient() override;

        std::vector<double> &getParameters() override;

    private:
        [[nodiscard]] std::vector<size_t> get_output_shape(const std::vector<size_t> &input_shape) const;

        const std::unique_ptr<IBlas> blas_;
        Tensor input_copy_;
        const size_t input_channels_;
        const size_t output_channels_;
        const size_t k1_;
        const size_t k2_;
        Tensor params_;
        Tensor grad_;
    };
}

#endif //MLP_CONV2D_H
