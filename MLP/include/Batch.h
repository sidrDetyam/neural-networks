//
// Created by sidr on 18.03.23.
//

#ifndef MLP_BATCH_H
#define MLP_BATCH_H

#include <memory>
#include <cstring>

class Batch {
public:

    Batch(size_t b_size, size_t feature_size) :
            b_size_(b_size),
            feature_size_(feature_size),
            ptr_(new double[b_size * feature_size]) {
    }

    Batch(const Batch &batch) :
            b_size_(batch.b_size_),
            feature_size_(batch.feature_size_) {

        auto *tmp = new double[b_size_ * feature_size_];
        memcpy(tmp, batch[0], b_size_ * feature_size_);
        ptr_ = std::unique_ptr<double>(tmp);
    }

    Batch(Batch &&batch)  noexcept :
            b_size_(batch.b_size_),
            feature_size_(batch.feature_size_),
            ptr_(batch.ptr_.get()){

        batch.ptr_.reset(nullptr);
    }

    virtual ~Batch() = default;

    double *operator[](size_t i) {
        return &ptr_.get()[i * feature_size_];
    }

    const double *operator[](size_t i) const {
        return &ptr_.get()[i * feature_size_];
    }

    [[nodiscard]] size_t getBsize() const {
        return b_size_;
    }

    [[nodiscard]] size_t getFeatureSize() const {
        return b_size_;
    }

private:
    std::unique_ptr<double> ptr_;
    const size_t b_size_;
    const size_t feature_size_;
};

#endif //MLP_BATCH_H
