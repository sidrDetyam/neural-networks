//
// Created by sidr on 15.04.23.
//

#ifndef TORCH4THEPOOREST_CACHINGDATALOADER_H
#define TORCH4THEPOOREST_CACHINGDATALOADER_H

#include <random>
#include "IDataLoader.h"

namespace nn {

    class CachingDataLoader : public IDataLoader {
    public:
        [[maybe_unused]] explicit CachingDataLoader(IDataLoader& dataLoader);

        [[maybe_unused]] explicit CachingDataLoader(IDataLoader&& dataLoader);

        [[maybe_unused]] [[maybe_unused]] explicit CachingDataLoader(std::vector<batch_t> data);

        batch_t next_batch() override;

        bool has_next() override;

        std::vector<batch_t> read_all() override;

        [[maybe_unused]] void shuffle();

        [[nodiscard]] size_t size() const;

        [[maybe_unused]] void reset();

    private:
        std::vector<batch_t> data_;
        size_t pos_;
        std::mt19937 generator_;
    };

}

#endif //TORCH4THEPOOREST_CACHINGDATALOADER_H
