//
// Created by sidr on 21.05.23.
//

#ifndef TORCH4THEPOOREST_RNNCELL_H
#define TORCH4THEPOOREST_RNNCELL_H

#include "IActivation.h"
#include "IBlas.h"
#include "Linear.h"

namespace nn {

    class RnnCell {
    public:
        explicit RnnCell(size_t input_size,
                         size_t hidden_size,
                         std::unique_ptr<IActivation> &&activation,
                         std::function<IBlas*()> &&blas_factory);

        void forward(Tensor &&input_tensor,
                     Tensor &&hidden_tensor,
                     const double* params);

        void backward(const Tensor &output,
                      double* grad);

        Tensor& get_input_grad();

        Tensor& get_hidden_grad();

    private:
        const size_t input_size_;
        const size_t hidden_size_;
        std::unique_ptr<IActivation> &&activation_;
        Linear dense_;
        std::function<IBlas*()> &&blas_factory_;
        Tensor output_;
        Tensor input_grad_;
        Tensor hidden_grad_;
    };
}

#endif //TORCH4THEPOOREST_RNNCELL_H
