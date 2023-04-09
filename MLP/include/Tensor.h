//
// Created by sidr on 18.03.23.
//

#ifndef MLP_TENSOR_H
#define MLP_TENSOR_H

#include <memory>
#include <cstring>
#include <cassert>
#include <vector>

using tdata_t = std::vector<double>;
using tshape_t = std::vector<size_t>;

class Tensor final {
public:
    Tensor() = default;

    explicit Tensor(tshape_t shape);

    explicit Tensor(tdata_t data, tshape_t shape);

    double *operator[](size_t i);

    const double *operator[](size_t i) const;

    [[nodiscard]] const tshape_t & get_shape() const;

    [[nodiscard]] size_t getBsize() const;

    [[nodiscard]] size_t getFeatureSize() const;

    [[nodiscard]] bool isSameShape(const Tensor& other) const;

    [[maybe_unused]] void reshape(tshape_t new_shape);

    [[nodiscard]] double* get_ptr(const std::vector<size_t>& coord);

    [[nodiscard]] const double* get_ptr(const std::vector<size_t>& coord) const;

    [[nodiscard]] tdata_t& data();

    [[nodiscard]] const tdata_t& data() const;

private:
    tshape_t shape_;
    tdata_t data_;
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

#endif //MLP_TENSOR_H
