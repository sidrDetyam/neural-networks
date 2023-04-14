//
// Created by sidr on 10.04.23.
//

#include "CsvDataLoader.h"
#include "Utils.h"

using namespace nn;

nn::CsvDataLoader::CsvDataLoader(const int batch_size,
                                 const bool last_incomplete,
                                 const std::string &fname,
                                 const size_t feature_size,
                                 std::vector<size_t> targets) :
        reader_(fname),
        last_incomplete_(last_incomplete),
        batch_size_(batch_size),
        target_cols_(std::move(targets)),
        feature_size_(feature_size){
    ASSERT_RE(batch_size_ > 0 && !target_cols_.empty() && last_incomplete && target_cols_.size() < feature_size);
    std::sort(target_cols_.begin(), target_cols_.end());
}

nn::batch_t nn::CsvDataLoader::next_batch() {

    std::vector<double> features;
    std::vector<double> targets;

    size_t cnt = 0;
    while(has_next() && cnt < batch_size_) {
        csv::CSVRow row;
        ASSERT_RE(reader_.read_row(row));
        ASSERT_RE(row.size() == feature_size_);

        size_t ind = 0;
        for (auto &field: row) {
            if(std::binary_search(target_cols_.begin(), target_cols_.end(), ind)){
                targets.push_back(field.get<double>());
            }
            else{
                features.push_back(field.get<double>());
            }
            ++ind;
        }
        ++cnt;
    }

    Tensor f(features, {cnt, feature_size_ - target_cols_.size()});
    Tensor t(targets, {cnt, target_cols_.size()});

    return {f, t};
}

bool nn::CsvDataLoader::has_next() {
    return !reader_.eof();
}

std::vector<batch_t> nn::CsvDataLoader::read_all() {

    std::vector<batch_t> data;

    while(has_next()){
        data.emplace_back(next_batch());
    }

    return data;
}
