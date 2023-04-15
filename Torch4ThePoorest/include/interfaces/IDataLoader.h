//
// Created by sidr on 15.04.23.
//

#ifndef TORCH4THEPOOREST_IDATALOADER_H
#define TORCH4THEPOOREST_IDATALOADER_H

#include "Tensor.h"

namespace nn{

    // Tensors 2x2 of features & targets
    using batch_t = std::pair<Tensor, Tensor>;

    class IDataLoader{
    public:
        virtual batch_t next_batch() = 0;

        virtual bool has_next() = 0;

        virtual std::vector<batch_t> read_all() = 0;

        virtual ~IDataLoader() = default;
    };
}

#endif //TORCH4THEPOOREST_IDATALOADER_H
