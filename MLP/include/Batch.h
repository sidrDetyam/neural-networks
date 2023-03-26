//
// Created by sidr on 18.03.23.
//

#ifndef MLP_BATCH_H
#define MLP_BATCH_H

#include <memory>
#include <cstring>
#include <cassert>
#include <vector>

class Batch final {
public:
    Batch();

    explicit Batch(size_t b_size, size_t feature_size);

    Batch(std::initializer_list<std::initializer_list<double>> b);

    double *operator[](size_t i);

    const double *operator[](size_t i) const;

    [[nodiscard]] size_t getBsize() const;

    [[nodiscard]] size_t getFeatureSize() const;

    [[nodiscard]] bool isSameShape(const Batch &other) const;

private:
    size_t b_size_;
    size_t feature_size_;
    std::vector<double> data_;
};


#endif //MLP_BATCH_H
