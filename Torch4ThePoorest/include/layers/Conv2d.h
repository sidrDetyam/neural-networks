//
// Created by sidr on 12.04.23.
//

#ifndef TORCH4THEPOOREST_CONV2D_H
#define TORCH4THEPOOREST_CONV2D_H

#include "ILayer.h"
#include "IBlas.h"

namespace nn {

    class Conv2d : public ILayer {
    public:
        explicit Conv2d(size_t input_channels,
                        size_t output_channels,
                        size_t kernel,
                        std::unique_ptr<IBlas> blas,
                        std::vector<double> params
        );

        Tensor forward(Tensor &&input) override;

        Tensor backward(const Tensor &output) override;

        std::vector<double> &getParametersGradient() override;

        std::vector<double> &getParameters() override;

        static void img2col(const double* original, size_t h, size_t w, size_t kernel, double* res);
    private:
        const std::unique_ptr<IBlas> blas_;
        Tensor input_copy_;
        std::vector<double> buff;
        const size_t input_channels_;
        const size_t output_channels_;
        const size_t kernel_;
        Tensor params_;
        Tensor grad_;
    };
}

#endif //TORCH4THEPOOREST_CONV2D_H
