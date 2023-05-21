//
// Created by sidr on 21.05.23.
//

#ifndef TORCH4THEPOOREST_RNNCELL_H
#define TORCH4THEPOOREST_RNNCELL_H

#include "IActivation.h"
#include "IBlas.h"

namespace nn {

    class RnnCell {
    public:
        explicit RnnCell(size_t input_size,
                         size_t hidden_size,
                         std::unique_ptr<IActivation> &&activation,
                         std::unique_ptr<IBlas> &&blas);

        void forward(Tensor &&input_tensor,
                     Tensor &&hidden_tensor,
                     );

        void backward(const Tensor &output);

    private:
        const size_t input_size_;
        const size_t output_size_;
        const size_t cnt_layers_;
        const size_t sequence_length_;
        std::function<IActivation * ()> activation_factory_;
        std::function<IBlas * ()> blas_factory_;
        std::vector <std::pair<Linear, std::unique_ptr < IActivation>>>
        cells_;
    };
}

#endif //TORCH4THEPOOREST_RNNCELL_H
