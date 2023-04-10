#include "AvgPolling.h"
#include "Utils.h"

//
// Created by sidr on 09.04.23.
//
Tensor AvgPolling::forward(Tensor &&input) {
    input_ = std::move(input);
    const auto &is = input_.get_shape();
    ASSERT_RE(is.size() == 4 && is[1] == channels_ && is[2] % k1_ == 0 && is[3] % k2_ == 0);

    const size_t out_height = is[2] / k1_;
    const size_t out_width = is[3] / k2_;

    Tensor output({is[0], is[1], out_height, out_width});

    for (size_t b = 0; b < is[0]; ++b) {
        for (size_t c = 0; c < is[1]; ++c) {
            for (size_t i = 0; i < out_height; ++i) {
                for (size_t j = 0; j < out_width; ++j) {
                    double sum = 0;
                    for (size_t p = 0; p < k1_; ++p) {
                        for (size_t q = 0; q < k2_; ++q) {
                            sum += input_({b, c, i * k1_ + p, j * k2_ + q});
                        }
                    }

                    output({b, c, i, j}) = sum / (double) (k1_ * k2_);
                }
            }
        }
    }

    return output;
}

Tensor AvgPolling::backward(const Tensor &output) {
    const auto &os = output.get_shape();
    const std::vector<size_t> is{os[0], os[1], os[2] * k1_, os[3] * k2_};
    ASSERT_RE(is == input_.get_shape());

    Tensor grad_input(is);

    for (size_t b = 0; b < os[0]; ++b) {
        for (size_t c = 0; c < os[1]; ++c) {
            for (size_t i = 0; i < os[2]; ++i) {
                for (size_t j = 0; j < os[3]; ++j) {
                    const double grad = output({b, c, i, j}) / (double) (k1_ * k2_);
                    for (size_t p = 0; p < k1_; ++p) {
                        for (size_t q = 0; q < k2_; ++q) {
                            grad_input({b, c, i * k1_ + p, j * k2_ + q}) = grad;
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

std::vector<double> &AvgPolling::getParametersGradient() {
    return empty_;
}

std::vector<double> &AvgPolling::getParameters() {
    return empty_;
}

AvgPolling::AvgPolling(size_t k1, size_t k2, size_t channels) :
        k1_(k1), k2_(k2), channels_(channels) {

}
