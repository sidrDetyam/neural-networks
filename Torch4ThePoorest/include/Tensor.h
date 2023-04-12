//
// Created by sidr on 18.03.23.
//

#ifndef MLP_TENSOR_H
#define MLP_TENSOR_H

#include <memory>
#include <cstring>
#include <cassert>
#include <vector>
#include <functional>

namespace nn {

    using tdata_t = std::vector<double>;
    using tshape_t = std::vector<size_t>;

    class Tensor final {
    public:
        Tensor() = default;

        explicit Tensor(tshape_t shape);

        explicit Tensor(tdata_t data, tshape_t shape);

        double *operator[](size_t i);

        const double *operator[](size_t i) const;

        [[nodiscard]] const tshape_t &get_shape() const;

        [[nodiscard]] size_t getBsize() const;

        [[nodiscard]] size_t getFeatureSize() const;

        [[nodiscard]] size_t size() const;

        [[nodiscard]] bool isSameShape(const Tensor &other) const;

        [[maybe_unused]] void reshape(tshape_t new_shape);

        void resize(tshape_t new_size);

        [[nodiscard]] double &operator()(const std::vector<size_t> &coord);

        [[nodiscard]] const double &operator()(const std::vector<size_t> &coord) const;

        [[nodiscard]] tdata_t &data();

        [[maybe_unused]] [[nodiscard]] const tdata_t &data() const;

        void map(const std::function<double(double)> &func);

    private:
        [[nodiscard]] size_t get_element_index(const std::vector<size_t> &coord) const;

        tshape_t shape_;
        tdata_t data_;
    };
}

std::ostream &operator<<(std::ostream &os, const nn::Tensor &tensor);

#endif //MLP_TENSOR_H
