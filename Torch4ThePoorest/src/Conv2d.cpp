#include "Conv2d.h"
#include "Utils.h"

//
// Created by sidr on 26.03.23.
//

using namespace nn;

Conv2d::Conv2d(const size_t input_channels,
               const size_t output_channels,
               const size_t k1,
               const size_t k2,
               std::unique_ptr<IBlas> blas,
               std::vector<double> params) :
        input_channels_(input_channels),
        output_channels_(output_channels),
        k1_(k1),
        k2_(k2),
        blas_(std::move(blas)) {

    ASSERT_RE(params.size() == output_channels_ * input_channels_ * k1_ * k2_);
    params_ = Tensor(std::move(params), {output_channels_, input_channels_, k1_, k2_});
    grad_ = Tensor({output_channels_, input_channels_, k1_, k2_});
}

Tensor Conv2d::forward(Tensor &&input) {

    input_copy_ = std::move(input);
    ASSERT_RE(input_copy_.get_shape()[1] == input_channels_);
    ASSERT_RE(input_copy_.get_shape()[2] >= k1_ && input_copy_.get_shape()[3] >= k2_);

    const std::vector<size_t> &input_shape = input_copy_.get_shape();
    const std::vector<size_t> output_shape = get_output_shape(input_shape);
    Tensor output(output_shape);

    for (size_t b = 0; b < output_shape[0]; ++b) {
        for (size_t out_c = 0; out_c < output_channels_; ++out_c) {
            for (size_t in_c = 0; in_c < input_channels_; ++in_c) {
                blas_->convolve(&input_copy_({b, in_c}),
                                &params_({out_c, in_c}),
                                &output({b, out_c}),
                                (int) input_shape[2], (int) input_shape[3],
                                (int) k1_, (int) k2_,
                                1.);
            }
        }
    }

    return output;
}

//TODO
//static double tmp_submatrix_sum(const double *const a,
//                                const size_t n,
//                                const size_t m,
//                                const size_t lda) {
//
//    double res = 0.;
//    for (size_t i = 0; i < n; ++i) {
//        for (size_t j = 0; j < m; ++j) {
//            res += a[i * lda + j];
//        }
//    }
//
//    return res;
//}

static double matrix_vector_ddot(const double *const a,
                                 const size_t n,
                                 const size_t m,
                                 const size_t lda,
                                 const double *const v) {
    double res = 0.;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            res += a[i * lda + j] * v[i * m + j];
        }
    }
    return res;
}

Tensor Conv2d::backward(const Tensor &output) {
    ASSERT_RE(output.get_shape() == get_output_shape(input_copy_.get_shape()));

    const auto &is = input_copy_.get_shape();
    const auto &os = output.get_shape();
    //const std::vector<double> e_((is[2] - k1_ + 1) * (is[3] - k2_ + 1), 1.);

    //bruuuh
    grad_.map([](double _) { return 0.; });

    for (size_t b = 0; b < is[0]; ++b) {
        for (size_t c_out = 0; c_out < output_channels_; ++c_out) {
            for (size_t c_in = 0; c_in < input_channels_; ++c_in) {
                double *const base_grad = &grad_({c_out, c_in});
                const double *const base_input = &input_copy_({b, c_in});
                const double *const base_output = &output({b, c_out});

                for (size_t k1 = 0; k1 < k1_; ++k1) {
                    for (size_t k2 = 0; k2 < k2_; ++k2) {
                        base_grad[k1 * k2_ + k2] += matrix_vector_ddot(base_input + is[3] * k1 + k2,
                                                                      os[2],
                                                                      os[3],
                                                                      is[3],
                                                                      base_output);
                    }
                }
            }
        }
    }
    blas_->scale(grad_[0], (int)grad_.size(), 1. / (double)is[0]);

    input_copy_.map([](double _){return 0.;});

    for(size_t b=0; b < is[0]; ++b) {
        for (size_t c_in = 0; c_in < input_channels_; ++c_in) {
            for (size_t c_out = 0; c_out < output_channels_; ++c_out) {
                double *const base_input = &input_copy_({b, c_in});
                const double *const base_output = &output({b, c_out});
                const double *const base_params = &params_({c_out, c_in});

                for (int i = 0; i < is[2]; ++i) {
                    for (int j = 0; j < is[3]; ++j) {

                        const int r_d = std::max(0, i - (int) k1_ + 1);
                        const int r_u = std::min(i, (int) is[2] - (int) k1_);
                        const int c_d = std::max(0, j - (int) k2_ + 1);
                        const int c_u = std::min(j, (int) is[3] - (int) k2_);

                        for (int r = r_d; r <= r_u; ++r) {
                            for (int c = c_d; c <= c_u; ++c) {
                                base_input[i*is[3] + j] += base_params[(i-r)*k2_ + (j-c)] * base_output[r*os[3] + c];
                            }
                        }
                    }
                }
            }
        }
    }

    return std::move(input_copy_);
}

std::vector<double> &Conv2d::getParametersGradient() {
    return grad_.data();
}

std::vector<double> &Conv2d::getParameters() {
    return params_.data();
}

std::vector<size_t> Conv2d::get_output_shape(const std::vector<size_t> &input_shape) const {
    ASSERT_RE(input_shape.size() == 4);
    ASSERT_RE(input_copy_.get_shape()[1] == input_channels_);
    ASSERT_RE(input_copy_.get_shape()[2] >= k1_ && input_copy_.get_shape()[3] >= k2_);

    std::vector<size_t> output_shape = input_shape;
    output_shape[1] = output_channels_;
    output_shape[2] = output_shape[2] - k1_ + 1;
    output_shape[3] = output_shape[3] - k2_ + 1;
    return output_shape;
}
