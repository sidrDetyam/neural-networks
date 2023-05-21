#include "Rnn.h"
#include "Utils.h"

//
// Created by sidr on 20.05.23.
//

using namespace nn;

Tensor nn::Rnn::forward(Tensor &&input) {
    const auto is = input.get_shape();
    ASSERT_RE(is.size() == 3 && is[1]==sequence_length_ && is[2]==input_size_);

    auto input_seq = input_sequence(std::move(input));
    auto init_h = std::vector<Tensor>(cnt_layers_, Tensor({is[0], output_size_}));

    for(size_t l=0; l<cnt_layers_; ++l){
        Tensor h_l = init_h[l];
        for(size_t t=0; t<sequence_length_; ++t){
            Tensor in = l==0? std::move(input_seq[t]) : cells_[l-1][t].get_output();

            cells_[l][t].forward(std::move(in), std::move(h_l), get_layer_param(l));
            h_l = cells_[l][t].get_output();
        }
    }

    return cells_[cnt_layers_-1][sequence_length_-1].get_output();
}

Tensor nn::Rnn::backward(const Tensor &output) {

    const auto& os = output.get_shape();
    ASSERT_RE(os.size() == 2 && os[2]==output_size_);

    for(int l=(int)cnt_layers_-1; l>=0; --l){
        for(int t=(int)sequence_length_-1; t>=0; --t){
            Tensor output_grad({os[0], output_size_});

            if(l != (int)cnt_layers_ -1){
                output_grad = element_wise_sum(cells_[l+1][t].get_input_grad(), output_grad);
            }

            if(t != (int)sequence_length_-1){
                output_grad = element_wise_sum(cells_[l][t+1].get_hidden_grad(), output_grad);
            }
            else{
                output_grad = element_wise_sum(output, output_grad);
            }

            cells_[l][t].backward(output_grad, get_layer_grad(l));
        }
    }

    return {};
}

Rnn::Rnn(const size_t input_size,
         const size_t output_size,
         const size_t cnt_layers,
         const size_t sequence_length,
         std::function<IActivation *()> activation_factory,
         std::function<IBlas *()> blas_factory) :
        input_size_(input_size),
        output_size_(output_size),
        cnt_layers_(cnt_layers),
        sequence_length_(sequence_length),
        activation_factory_(std::move(activation_factory)),
        blas_factory_(std::move(blas_factory)),
        cells_(cnt_layers){

    ASSERT_RE(input_size > 0 && output_size > 0 && cnt_layers_ > 0 && sequence_length_ > 0);

    params_.assign((input_size_ + output_size_) * output_size_ +
                           (output_size_+output_size_) * output_size_ * (cnt_layers_-1), 0);
    grad_.assign(params_.size(), 0);

    for (size_t i = 0; i < cnt_layers_; ++i) {
        const size_t cell_input_size = i==0? input_size_ : output_size_;

        for(size_t j=0; j < sequence_length_; ++j) {
            auto activation = std::unique_ptr<IActivation>(activation_factory_());
            std::function<IBlas *()> blas_factory_cell = [this]() {
                return blas_factory_();
            };
            cells_[i].emplace_back(cell_input_size, output_size_,
                                std::move(activation), std::move(blas_factory_cell));
        }
    }

    blas_ = std::unique_ptr<IBlas>(blas_factory_());
}

size_t Rnn::get_layers_element_ind(const size_t layer_ind) const {
    if(layer_ind == 0){
        return 0;
    }

    return (input_size_+output_size_)*output_size_ + (layer_ind-1) * (output_size_+output_size_)*output_size_;
}

double *Rnn::get_layer_param(const size_t layer_ind) {
    return &params_[get_layers_element_ind(layer_ind)];
}

double *Rnn::get_layer_grad(const size_t layer_ind) {
    return &grad_[get_layers_element_ind(layer_ind)];
}

std::vector<Tensor> Rnn::input_sequence(Tensor &&input) const{
    std::vector<Tensor> res;

    const auto& is = input.get_shape();
    ASSERT_RE(is.size()==3 && is[1]==sequence_length_ && is[2]==input_size_);

    for(size_t i=0; i<sequence_length_; ++i){
        Tensor seq_element({is[0], input_size_});
        for(size_t b=0; b<is[0]; ++b){
            std::memcpy(&seq_element({b}), &input({b, i}), sizeof(double)*input_size_);
        }
        res.emplace_back(std::move(seq_element));
    }

    return res;
}

Tensor Rnn::element_wise_sum(const Tensor &a, const Tensor &b) {
    ASSERT_RE(a.isSameShape(b));
    Tensor res(a.get_shape());
    blas_->daxpby((int) res.data().size(), a.data().data(), 1., res.data().data(), 1.);
    blas_->daxpby((int) res.data().size(), b.data().data(), 1., res.data().data(), 1.);
    return res;
}

