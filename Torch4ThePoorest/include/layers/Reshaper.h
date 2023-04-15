//
// Created by sidr on 10.04.23.
//

#ifndef TORCH4THEPOOREST_RESHAPER_H
#define TORCH4THEPOOREST_RESHAPER_H

#include "ILayer.h"

namespace nn{

    class Reshaper: public ILayer{
    public:
        explicit Reshaper(tshape_t is_, tshape_t os_);

        Tensor forward(Tensor &&input) override;

        Tensor backward(const Tensor &output) override;

    private:
        const tshape_t is_;
        const tshape_t os_;
    };

}

#endif //TORCH4THEPOOREST_RESHAPER_H
