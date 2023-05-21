//
// Created by sidr on 20.05.23.
//

#ifndef TORCH4THEPOOREST_RNN_H
#define TORCH4THEPOOREST_RNN_H

#include "ILayer.h"
#include "IActivation.h"
#include "Linear.h"
#include <functional>

namespace nn {

    class Rnn: ILayer {
    public:
        explicit Rnn(size_t input_size_,
                     size_t output_size_,
                     size_t cnt_layers_,
                     size_t sequence_length_,
                     std::function<IActivation*()> activation_factory,
                     std::function<IBlas*()> blas_factory);

        Tensor forward(Tensor &&input) override;

        Tensor backward(const Tensor &output) override;

    private:
        const size_t input_size_;
        const size_t output_size_;
        const size_t cnt_layers_;
        const size_t sequence_length_;
        std::function<IActivation*()> activation_factory_;
        std::function<IBlas*()> blas_factory_;
        std::vector<std::pair<Linear, std::unique_ptr<IActivation>>> cells_;
    };
}

#endif //TORCH4THEPOOREST_RNN_H
