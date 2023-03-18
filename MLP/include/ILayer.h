//
// Created by sidr on 18.03.23.
//

#ifndef MLP_ILAYER_H
#define MLP_ILAYER_H

#include "Batch.h"

class ILayer {
public:

    virtual Batch forward(const Batch& input) = 0;

    virtual Batch backward(const Batch& output) = 0;

    virtual ~ILayer() = default;
};

#endif //MLP_ILAYER_H
