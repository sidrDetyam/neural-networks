//
// Created by sidr on 18.03.23.
//

#ifndef MLP_LINEARLAYER_H
#define MLP_LINEARLAYER_H

#include "ILayer.h"
#include "IBlas.h"
#include <memory>

class LinearLayer: public ILayer{
public:
    explicit LinearLayer(size_t input_size, size_t output_size, std::unique_ptr<IBlas> &&blas);

    Batch forward(const Batch &input) override;

    Batch backward(const Batch &output) override;

private:
    std::unique_ptr<IBlas> blas_;
    const size_t input_size_;
    const size_t output_size_;
};

#endif //MLP_LINEARLAYER_H
