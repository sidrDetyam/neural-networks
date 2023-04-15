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
                        std::vector<double> params,
                        bool bias = false
        );

        Tensor forward(Tensor &&input) override;

        Tensor backward(const Tensor &output) override;

        static void add_padding(const double* source, double* dest, size_t h, size_t w, size_t l, size_t t, size_t r, size_t b);

    private:
        [[nodiscard]] std::vector<size_t> get_output_shape(const std::vector<size_t> &input_shape) const;

        static void img2col(const double *original,
                            size_t h, size_t w,
                            size_t kernel1, size_t kernel2,
                            double *res,
                            size_t lda = 0);

        void calculate_params_grad(const Tensor &output);

        [[nodiscard]] double &get_bias_part_param();

        [[nodiscard]] double &get_bias_part_grad();

        const std::unique_ptr<IBlas> blas_;
        Tensor input_copy_;
        Tensor buff_;
        const size_t input_channels_;
        const size_t output_channels_;
        const size_t kernel_;
        bool bias_;

        Tensor im2col_buff_;
        Tensor input_grads_shuffled_;
    };
}

#endif //TORCH4THEPOOREST_CONV2D_H
