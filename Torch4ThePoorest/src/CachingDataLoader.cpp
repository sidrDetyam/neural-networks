//
// Created by sidr on 15.04.23.
//

#include "CachingDataLoader.h"
#include "Utils.h"

nn::batch_t nn::CachingDataLoader::next_batch() {
    ASSERT_RE(has_next());
    auto batch = data_[pos_];
    ++pos_;
    return batch;
}

bool nn::CachingDataLoader::has_next() {
    return pos_ < data_.size();
}

std::vector<nn::batch_t> nn::CachingDataLoader::read_all() {
    std::vector<nn::batch_t> rest;
    rest.resize(data_.size() - pos_);
    std::copy_n(data_.cbegin(), data_.size() - pos_, rest.begin());

    return rest;
}

[[maybe_unused]] nn::CachingDataLoader::CachingDataLoader(nn::IDataLoader &dataLoader) {
    while(dataLoader.has_next()){
        batch_t batch = dataLoader.next_batch();
        data_.emplace_back(std::move(batch));
    }
    pos_ = 0;
    std::random_device rd;
    generator_ = std::mt19937(rd());
}

[[maybe_unused]] void nn::CachingDataLoader::shuffle() {
    pos_ = 0;
    std::shuffle(data_.begin(), data_.end(), generator_);
}

size_t nn::CachingDataLoader::size() const {
    return data_.size();
}

[[maybe_unused]] void nn::CachingDataLoader::reset() {
    pos_ = 0;
}

nn::CachingDataLoader::CachingDataLoader(nn::IDataLoader &&dataLoader): CachingDataLoader(dataLoader) {

}
