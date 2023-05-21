//
// Created by sidr on 20.05.23.
//

#ifndef TORCH4THEPOOREST_RNN_H
#define TORCH4THEPOOREST_RNN_H

#include "ILayer.h"
#include "IActivation.h"
#include "Linear.h"
#include "RnnCell.h"
#include <functional>

namespace nn {

    class Rnn: ILayer {
    public:
        explicit Rnn(size_t input_size,
                     size_t output_size,
                     size_t cnt_layers,
                     size_t sequence_length,
                     std::function<IActivation*()> activation_factory,
                     std::function<IBlas*()> blas_factory);

        Tensor forward(Tensor &&input) override;

        Tensor backward(const Tensor &output) override;

    private:
        [[nodiscard]] size_t get_layers_element_ind(size_t layer_ind) const;

        double* get_layer_param(size_t layer_ind);

        double* get_layer_grad(size_t layer_ind);

        std::vector<Tensor> input_sequence(Tensor &&input) const;

        Tensor element_wise_sum(const Tensor& a, const Tensor& b);

        const size_t input_size_;
        const size_t output_size_;
        const size_t cnt_layers_;
        const size_t sequence_length_;
        std::function<IActivation*()> activation_factory_;
        std::function<IBlas*()> blas_factory_;
        std::vector<std::vector<RnnCell>> cells_;
        std::unique_ptr<IBlas> blas_;
    };
}

#endif //TORCH4THEPOOREST_RNN_H
