//
// Created by sidr on 18.03.23.
//
#include "Tensor.h"
#include "Utils.h"

using namespace nn;

Tensor::Tensor(tdata_t data, tshape_t shape) :
        data_(std::move(data)),
        shape_(std::move(shape)) {

    for (auto i: shape_) {
        ASSERT_RE(i > 0);
    }

    ASSERT_RE(data_.size() == getBsize() * getFeatureSize());
}

Tensor::Tensor(tshape_t shape) : shape_(std::move(shape)) {
    for (auto i: shape_) {
        ASSERT_RE(i > 0);
    }
    data_.assign(size(), 0);
    ASSERT_RE(data_.size() == size());
}

double *Tensor::operator[](size_t i) {
    return &data_[i * getFeatureSize()];
}

const double *Tensor::operator[](size_t i) const {
    return &data_[i * getFeatureSize()];
}

size_t Tensor::getBsize() const {
    return shape_.empty() ? 0 : shape_[0];
}

size_t Tensor::getFeatureSize() const {
    ASSERT_RE(shape_.size() >= 2);

    size_t size = 1;
    for (size_t i = 1; i < shape_.size(); ++i) {
        size *= shape_[i];
    }
    return size;
}

const std::vector<size_t> &Tensor::get_shape() const {
    return shape_;
}

bool Tensor::isSameShape(const Tensor &other) const {
    return shape_ == other.shape_;
}

[[maybe_unused]] void Tensor::reshape(std::vector<size_t> new_shape) {

    size_t cnt = new_shape.empty() ? 0 : 1;
    for (auto i: new_shape) {
        ASSERT_RE(i >= 1);
        cnt *= i;
    }
    ASSERT_RE(cnt == getBsize() * getFeatureSize());
    shape_ = std::move(new_shape);
}

tdata_t &Tensor::data() {
    return data_;
}

[[maybe_unused]] const tdata_t &Tensor::data() const {
    return data_;
}

void Tensor::map(const std::function<double(double)> &func) {
    std::for_each(data_.begin(), data_.end(),
                  [&func](double &i) { i = func(i); });
}

size_t Tensor::size() const {
    size_t sz = !shape_.empty();
    for(auto dim : shape_){
        sz *= dim;
    }
    return sz;
}

size_t Tensor::get_element_index(const std::vector<size_t> &coord) const {
    ASSERT_RE(!coord.empty() && coord.size() <= shape_.size());
    size_t ind = 0;
    size_t sz = shape_.size()==1? 1 : getFeatureSize();

    for (size_t i = 0; i < coord.size(); ++i) {
        ind += sz * coord[i];
        if (i + 1 != coord.size()) {
            sz /= shape_[i + 1];
        }
    }

    return ind;
}

const double &Tensor::operator()(const std::vector<size_t> &coord) const {
    return data_[get_element_index(coord)];
}

double &Tensor::operator()(const std::vector<size_t> &coord) {
    return data_[get_element_index(coord)];
}

static void helper(std::ostream &os, const Tensor &tensor, const std::vector<size_t> &coord = {}, size_t dim = 0) {
    if (dim + 1 == tensor.get_shape().size()) {
        for (size_t i = 0; i < tensor.get_shape()[dim]; ++i) {
            auto coord_ = coord;
            coord_.push_back(i);
            os << tensor(coord_);
            if (i + 1 != tensor.get_shape()[dim]) {
                os << ", ";
            }
        }

        return;
    }

    for (size_t i = 0; i < tensor.get_shape()[dim]; ++i) {
        auto coord_ = coord;
        coord_.push_back(i);
        os << std::endl;
        for (size_t s = 0; s < dim; ++s) {
            os << "  ";
        }
        os << "[";
        helper(os, tensor, coord_, dim + 1);
        if (dim + 2 != tensor.get_shape().size()) {
            for (size_t s = 0; s < dim; ++s) {
                os << "  ";
            }
        }
        os << "], " << std::endl;
    }
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    os << "Tensor(";
    for(size_t i : tensor.get_shape()){
        os << i << ", ";
    }
    os << ")";
    helper(os, tensor);
    return os;
}
