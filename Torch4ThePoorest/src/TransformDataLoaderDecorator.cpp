//
// Created by sidr on 16.04.23.
//
#include "TransformDataLoaderDecorator.h"

nn::batch_t nn::TransformDataLoaderDecorator::next_batch() {
    ASSERT_RE(has_next());
    auto batch = wrapee_.next_batch();
    cb_(batch);
    return batch;
}

bool nn::TransformDataLoaderDecorator::has_next() {
    return wrapee_.has_next();
}

std::vector<nn::batch_t> nn::TransformDataLoaderDecorator::read_all() {
    auto batches = wrapee_.read_all();
    std::for_each(batches.begin(), batches.end(), cb_);
    return batches;
}

nn::TransformDataLoaderDecorator::TransformDataLoaderDecorator(nn::IDataLoader &wrapee,
                                                               const std::function<void(batch_t &)> &cb):
                                                               wrapee_(wrapee),
                                                               cb_(cb){
}
