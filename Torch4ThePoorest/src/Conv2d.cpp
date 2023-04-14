//
// Created by sidr on 12.04.23.
//
#include "Conv2d.h"
#include "Utils.h"

using namespace nn;

//nn::Tensor nn::Conv2d::forward(nn::Tensor &&input) {
//
//    input_copy_ = std::move(input);
//    ASSERT_RE(input_copy_.get_shape()[1] == input_channels_);
//    ASSERT_RE(input_copy_.get_shape()[2] >= kernel_ && input_copy_.get_shape()[3] >= kernel_);
//
//    const std::vector<size_t> &input_shape = input_copy_.get_shape();
//    const std::vector<size_t> output_shape = get_output_shape(input_shape);
//    Tensor output(output_shape);
//
//    buff_.resize({input_channels_, kernel_ * kernel_, output_shape[2] * output_shape[3]});
//
//    for (size_t b = 0; b < input_shape[0]; ++b) {
//        for (size_t c_in = 0; c_in < input_channels_; ++c_in) {
//            img2col(&input_copy_({b, c_in}),
//                    input_shape[2], input_shape[3], kernel_,
//                    &buff_({c_in}));
//        }
//
//        for (size_t c_out = 0; c_out < output_channels_; ++c_out) {
//            for (size_t c_in = 0; c_in < input_channels_; ++c_in) {
//                blas_->dgemm_full(ROW_ORDER, NO_TRANS, NO_TRANS,
//                                  (int) 1, (int) (output_shape[2] * output_shape[3]),
//                                  (int) (kernel_ * kernel_),
//                                  1., &params_({c_out, c_in}), (int) (kernel_ * kernel_),
//                                  &buff_({c_in}), (int) (output_shape[2] * output_shape[3]), 1.,
//                                  &output({b, c_out}), (int) (output_shape[2] * output_shape[3]));
//            }
//        }
//    }
//
//    return output;
//}

nn::Tensor nn::Conv2d::forward(nn::Tensor &&input) {

    input_copy_ = std::move(input);
    ASSERT_RE(input_copy_.get_shape()[1] == input_channels_);
    ASSERT_RE(input_copy_.get_shape()[2] >= kernel_ && input_copy_.get_shape()[3] >= kernel_);

    const std::vector<size_t> &input_shape = input_copy_.get_shape();
    const std::vector<size_t> output_shape = get_output_shape(input_shape);
    Tensor output(output_shape);

    buff_.resize({input_channels_, kernel_ * kernel_, output_shape[2] * output_shape[3]});

    for (size_t b = 0; b < input_shape[0]; ++b) {
        for (size_t c_in = 0; c_in < input_channels_; ++c_in) {
            img2col(&input_copy_({b, c_in}),
                    input_shape[2], input_shape[3], kernel_, kernel_,
                    &buff_({c_in}));
        }

        for (size_t c_out = 0; c_out < output_channels_; ++c_out) {
            blas_->dgemm_full(ROW_ORDER, NO_TRANS, NO_TRANS,
                              1, (int) (output_shape[2] * output_shape[3]),
                              (int) (kernel_ * kernel_ * input_channels_),
                              1., &params_[c_out * (input_channels_ * kernel_ * kernel_)],
                              (int) (kernel_ * kernel_ * input_channels_),
                              &buff_({0}), (int) (output_shape[2] * output_shape[3]), 0.,
                              &output({b, c_out}), (int) (output_shape[2] * output_shape[3]));
        }
    }

    if(bias_) {
        std::vector<size_t> coord(2);
        for (size_t b = 0; b < output_shape[0]; ++b) {
            for (size_t c_out = 0; c_out < output_channels_; ++c_out) {
                coord[0] = b;
                coord[1] = c_out;
                double *const base_ptr = &output(coord);
                double bias = (&get_bias_part_param())[c_out];
                blas_->daxpby_full((int) (output_shape[2] * output_shape[3]), &bias, 1., 0, base_ptr, 1., 1);
            }
        }
    }

    return output;
}

void Conv2d::calculate_params_grad(const Tensor &output) {
    ASSERT_RE(output.get_shape() == get_output_shape(input_copy_.get_shape()));

    const auto &is = input_copy_.get_shape();
    const auto &os = output.get_shape();

    buff_.resize({input_channels_, os[2] * os[3], (is[2] - os[2] + 1) * (is[3] - os[3] + 1)});

    Tensor kernels_grad(std::vector<double>(output_channels_*input_channels_*kernel_*kernel_),
                       {output_channels_, input_channels_, kernel_, kernel_});

    for (size_t b = 0; b < is[0]; ++b) {
        for (size_t c_in = 0; c_in < input_channels_; ++c_in) {
            img2col(&input_copy_({b, c_in}),
                    is[2], is[3], os[2], os[3],
                    &buff_({c_in}));
        }

        for (size_t c_out = 0; c_out < output_channels_; ++c_out) {
            const int m = 1;
            const int n = (int) (kernel_ * kernel_);
            const int k = (int) (os[2] * os[3]);

            for (size_t c_in = 0; c_in < input_channels_; ++c_in) {
                blas_->dgemm_full(ROW_ORDER, NO_TRANS, NO_TRANS,
                                  m, n, k,
                                  1., &output({b, c_out}), k,
                                  &buff_({c_in}), n, 1.,
                                  &kernels_grad({c_out, c_in}), n);
            }
        }
    }
    std::copy(kernels_grad.data().begin(), kernels_grad.data().end(), grad_.begin());

    if(bias_) {
        std::vector<size_t> coord(2);
        for (size_t b = 0; b < os[0]; ++b) {
            for (size_t c_out = 0; c_out < output_channels_; ++c_out) {
                coord[0] = b;
                coord[1] = c_out;
                const double *const base_ptr = &output(coord);
                double *const bias = &(&get_bias_part_grad())[c_out];
                blas_->daxpby_full((int) (os[2] * os[3]), base_ptr, 1., 1, bias, 1., 0);
            }
        }
    }

    blas_->scale(&grad_[0], (int) grad_.size(), 1. / (double) is[0]);
}


nn::Tensor nn::Conv2d::backward(const nn::Tensor &output) {
    ASSERT_RE(output.get_shape() == get_output_shape(input_copy_.get_shape()));
    calculate_params_grad(output);

    const auto &is = input_copy_.get_shape();
    const auto &os = output.get_shape();

    std::vector<double> raw_kernels(params_.data(), &get_bias_part_param());
    const Tensor kernels(std::move(raw_kernels), {output_channels_, input_channels_, kernel_, kernel_});
    Tensor reversed_kernels({input_channels_, output_channels_, kernel_, kernel_});

    for (size_t c_in = 0; c_in < input_channels_; ++c_in) {
        for (size_t c_out = 0; c_out < output_channels_; ++c_out) {
            double *const ptr = &reversed_kernels({c_in, c_out});
            std::copy_n(&kernels({c_out, c_in}), kernel_ * kernel_, ptr);
            std::reverse(ptr, ptr + kernel_ * kernel_);
        }
    }

    Tensor padding_buff({os[2] + 2 * kernel_ - 2, os[3] + 2 * kernel_ - 2});
    const size_t padding = kernel_ - 1;
    im2col_buff_.resize({output_channels_, kernel_ * kernel_, is[0], is[2] * is[3]});
    input_grads_shuffled_.resize({input_channels_, is[0], is[2] * is[3]});


    for (size_t b = 0; b < is[0]; ++b) {
        for (size_t c_out = 0; c_out < output_channels_; ++c_out) {
            add_padding(&output({b, c_out}), padding_buff.data().data(), os[2], os[3],
                        padding, padding, padding, padding);

            img2col(padding_buff.data().data(), is[2] + kernel_ - 1, is[3] + kernel_ - 1, kernel_, kernel_,
                    &im2col_buff_({c_out, 0, b}),
                    is[0] * is[2] * is[3]);
        }
    }

    const int m = (int) input_channels_;
    const int n = (int) (is[0] * is[2] * is[3]);
    const int k = (int) (kernel_ * kernel_ * output_channels_);

    blas_->dgemm_full(ROW_ORDER, NO_TRANS, NO_TRANS,
                      m, n, k,
                      1., reversed_kernels.data().data(), k,
                      im2col_buff_.data().data(), n, 0,
                      &input_grads_shuffled_({0}), n);

    for (size_t b = 0; b < is[0]; ++b) {
        for (size_t c_in = 0; c_in < input_channels_; ++c_in) {
            std::copy_n(&input_grads_shuffled_({c_in, b}), is[2] * is[3], &input_copy_({b, c_in}));
        }
    }

    return std::move(input_copy_);
}

std::vector<double> &nn::Conv2d::getParametersGradient() {
    return grad_;
}

std::vector<double> &nn::Conv2d::getParameters() {
    return params_;
}

Conv2d::Conv2d(const size_t input_channels,
               const size_t output_channels,
               const size_t kernel,
               std::unique_ptr<IBlas> blas,
               std::vector<double> params,
               const bool bias) :
        input_channels_(input_channels),
        output_channels_(output_channels),
        kernel_(kernel),
        blas_(std::move(blas)),
        bias_(bias) {

    ASSERT_RE(params.size() == output_channels_ * input_channels_ * kernel_ * kernel_ + (bias? output_channels_ : 0));
    params_ = std::move(params);
    grad_.assign(params_.size(), 0);
}

void Conv2d::img2col(const double *const original,
                     const size_t h, const size_t w,
                     const size_t kernel1,
                     const size_t kernel2,
                     double *const res,
                     size_t lda) {
    ASSERT_RE(h >= kernel1 && w >= kernel2);
    ASSERT_RE(lda >= (h - kernel1 + 1) * (w - kernel2 + 1) || lda == 0);
    if (lda == 0) {
        lda = (h - kernel1 + 1) * (w - kernel2 + 1);
    }

    //bruh
    for (size_t kh = 0; kh < kernel1; ++kh) {
        for (size_t kw = 0; kw < kernel2; ++kw) {
            for (size_t i = 0; i < h - kernel1 + 1; ++i) {
                for (size_t j = 0; j < w - kernel2 + 1; ++j) {
                    res[(kh * kernel2 + kw) * lda +
                        (i * (w - kernel2 + 1) + j)] = original[(i + kh) * (w) + (j + kw)];
                }
            }
        }
    }
}

std::vector<size_t> Conv2d::get_output_shape(const std::vector<size_t> &input_shape) const {
    ASSERT_RE(input_shape.size() == 4);
    ASSERT_RE(input_copy_.get_shape()[1] == input_channels_);
    ASSERT_RE(input_copy_.get_shape()[2] >= kernel_ && input_copy_.get_shape()[3] >= kernel_);

    std::vector<size_t> output_shape = input_shape;
    output_shape[1] = output_channels_;
    output_shape[2] = output_shape[2] - kernel_ + 1;
    output_shape[3] = output_shape[3] - kernel_ + 1;
    return output_shape;
}

void
Conv2d::add_padding(const double *source, double *dest, size_t h, size_t w, size_t l, size_t t, size_t r, size_t b) {
    const size_t ph = h + t + b;
    const size_t pw = w + l + r;
    for (size_t i = 0, si = 0; i < ph * pw; ++i) {
        const size_t y = i / pw;
        const size_t x = i - y * pw;
        if (y < t || h + t <= y || x < l || w + l <= x) {
            dest[i] = 0;
        } else {
            dest[i] = source[si];
            ++si;
        }
    }
}

double &Conv2d::get_bias_part_param(){
    return params_[output_channels_ * input_channels_ * kernel_ * kernel_];
}

double &Conv2d::get_bias_part_grad() {
    return grad_[output_channels_ * input_channels_ * kernel_ * kernel_];
}
