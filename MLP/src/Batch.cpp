//
// Created by sidr on 18.03.23.
//
#include "Batch.h"

Batch::Batch(): Batch(0, 0) {

}

bool Batch::isSameShape(const Batch &other) const {
    return feature_size_ == other.feature_size_ && b_size_ == other.b_size_;
}

