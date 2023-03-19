//
// Created by sidr on 18.03.23.
//

#ifndef MLP_BATCH_H
#define MLP_BATCH_H

#include <memory>
#include <cstring>
#include <cassert>

class Batch final{
public:

    Batch(size_t b_size, size_t feature_size) :
            b_size_(b_size),
            feature_size_(feature_size),
            ptr_(b_size == 0 && feature_size == 0? nullptr : new double[b_size * feature_size]) {

    }

    Batch(std::initializer_list<std::initializer_list<double>> b): Batch(b.size(), b.begin()->size()){

        auto iti = b.begin();
        for(size_t i=0; i<b_size_; ++i){
            auto itj = iti->begin();
            for(size_t j=0; j<feature_size_; ++j){
                ptr_.get()[i*feature_size_ + j] = *itj;
                ++itj;
            }
            ++iti;
        }
    }

    Batch(const Batch &batch) :
            b_size_(batch.b_size_),
            feature_size_(batch.feature_size_) {

        assert(b_size_ >= 0 && feature_size_ >= 0);

        auto *tmp = new double[b_size_ * feature_size_];
        memcpy(tmp, batch[0], b_size_ * feature_size_ * sizeof(double));
        ptr_ = std::unique_ptr<double>(tmp);
    }

    Batch& operator =(const Batch &batch){
        if(this == &batch){
            return *this;
        }

        if(b_size_ != batch.b_size_ || feature_size_ != batch.feature_size_){
            ptr_.reset(new double[batch.b_size_ * batch.feature_size_]);
        }

        b_size_ = batch.b_size_;
        feature_size_ = batch.feature_size_;
        memcpy(ptr_.get(), batch[0], b_size_ * feature_size_ * sizeof(double));
        return *this;
    }

    Batch& operator =(Batch &&batch) noexcept {
        b_size_ = batch.b_size_;
        feature_size_ = batch.feature_size_;
        batch.b_size_ = 0;
        batch.feature_size_ = 0;
        ptr_ = std::move(batch.ptr_);
        return *this;
    }

    Batch(Batch &&batch)  noexcept :
            b_size_(batch.b_size_),
            feature_size_(batch.feature_size_),
            ptr_(std::move(batch.ptr_)){
        batch.b_size_ = 0;
        batch.feature_size_ = 0;
    }

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
        return feature_size_;
    }

private:
    std::unique_ptr<double> ptr_;
    size_t b_size_;
    size_t feature_size_;
};

using Matrix = Batch;

#endif //MLP_BATCH_H
