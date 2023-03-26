//
// Created by sidr on 18.03.23.
//
#include "Batch.h"

Batch::Batch() : Batch(0, 0) {

}

bool Batch::isSameShape(const Batch &other) const {
    return feature_size_ == other.feature_size_ && b_size_ == other.b_size_;
}

Batch::Batch(size_t b_size, size_t feature_size) :
        b_size_(b_size),
        feature_size_(feature_size),
        data_(b_size_ * feature_size_) {
}

Batch::Batch(std::initializer_list<std::initializer_list<double>> b) : Batch(b.size(), b.begin()->size()) {

    auto iti = b.begin();
    for (size_t i = 0; i < b_size_; ++i) {
        auto itj = iti->begin();
        for (size_t j = 0; j < feature_size_; ++j) {
            data_[i * feature_size_ + j] = *itj;
            ++itj;
        }
        ++iti;
    }
}

double *Batch::operator[](size_t i) {
    return &data_[i * feature_size_];
}

const double *Batch::operator[](size_t i) const {
    return &data_[i * feature_size_];
}

size_t Batch::getBsize() const {
    return b_size_;
}

size_t Batch::getFeatureSize() const {
    return feature_size_;
}

