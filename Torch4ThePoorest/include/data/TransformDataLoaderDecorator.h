//
// Created by sidr on 16.04.23.
//

#ifndef TORCH4THEPOOREST_TRANSFORMDATALOADERDECORATOR_H
#define TORCH4THEPOOREST_TRANSFORMDATALOADERDECORATOR_H

#include "IDataLoader.h"
#include "Utils.h"
#include <functional>

namespace nn {

    class TransformDataLoaderDecorator: public IDataLoader {
    public:
        explicit TransformDataLoaderDecorator(IDataLoader& wrapee,
                                              const std::function<void(batch_t&)> &cb);

        batch_t next_batch() override;

        bool has_next() override;

        std::vector<batch_t> read_all() override;

    private:
        IDataLoader& wrapee_;
        const std::function<void(batch_t&)> &cb_;
    };

}

#endif //TORCH4THEPOOREST_TRANSFORMDATALOADERDECORATOR_H
