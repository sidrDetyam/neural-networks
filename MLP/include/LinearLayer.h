//
// Created by sidr on 18.03.23.
//

#ifndef MLP_LINEARLAYER_H
#define MLP_LINEARLAYER_H

#include "ILayer.h"
#include "IBlas.h"
#include <memory>
#include <vector>

class LinearLayer: public ILayer{
public:
    explicit LinearLayer(size_t input_size, size_t output_size, std::unique_ptr<IBlas> &&blas);

    const Batch& forward(const Batch &input) override;

    Batch backward(const Batch &output) override;

    Matrix& getWeights(){
        return weights_;
    }

    Matrix& getBias(){
        return bias_;
    }

private:
    std::unique_ptr<IBlas> blas_;
    const size_t input_size_;
    const size_t output_size_;

    Matrix weights_;
    Matrix w_grad_;
    Matrix bias_;
    Matrix b_grad_;
    Batch input_;
    Batch output_;
};

#endif //MLP_LINEARLAYER_H
