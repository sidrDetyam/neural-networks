//
// Created by sidr on 18.03.23.
//

#ifndef MLP_LINEARLAYER_H
#define MLP_LINEARLAYER_H

#include "ILayer.h"
#include "IBlas.h"
#include <vector>

namespace nn {

    class Linear : public ILayer {
    public:
        explicit Linear(size_t input_size,
                        size_t output_size,
                        std::vector<double> weights,
                        std::vector<double> bias,
                        std::unique_ptr<IBlas> blas);

        explicit Linear(size_t input_size,
                        size_t output_size,
                        std::unique_ptr<IBlas> blas);

        Tensor forward(Tensor &&input) override;

        Tensor backward(const Tensor &output) override;

    private:
        double *getBPart();

        double *getGradBPart();

        std::unique_ptr<IBlas> blas_;
        const size_t input_size_;
        const size_t output_size_;

        Tensor input_;
    };
}

#endif //MLP_LINEARLAYER_H
