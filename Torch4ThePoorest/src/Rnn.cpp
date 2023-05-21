#include "Rnn.h"
#include "Utils.h"

//
// Created by sidr on 20.05.23.
//

using namespace nn;

Tensor nn::Rnn::forward(Tensor &&input) {
    ASSERT_RE(input.get_shape().size() == 3 && input.get_shape()[1]==sequence_length_
        && input.get_shape()[2]==input_size_);

//    Tensor h({input.get_shape()[0], output_size_});
    const size_t cell_params_cnt = (input_size_ + output_size_) * output_size_;
    std::vector<Tensor> hidden(cnt_layers_, Tensor({input.get_shape()[0], output_size_}));

    for(size_t i=0; i<cnt_layers_; ++i){
        for(size_t j=0; j<sequence_length_; ++j){
            Tensor in({input.get_shape()[0], output_size_});
            for(size_t b=0; b<input.get_shape()[0]; ++b){
                std::memcpy(&in({b}), &input({b}), sizeof(double)*input_size_);
                std::memcpy(&in({b, input_size_}), &hidden[i]({b}), sizeof(double)*output_size_);
            }
            std::memcpy(cells_[i*j].first.getParameters().data(), &params_[cell_params_cnt*i], sizeof(double)*cell_params_cnt);
            cells_[i*j].first.getParametersGradient().assign(cells_[i*j].first.getParametersGradient().size(), 0);

            Tensor out_cell = cells_[i*j].first.forward(std::move(in));
            out_cell = cells_[i*j].second->forward(std::move(out_cell));

        }
    }

}

Tensor nn::Rnn::backward(const Tensor &output) {
    return Tensor();
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
        blas_factory_(std::move(blas_factory)) {

    ASSERT_RE(input_size > 0 && output_size > 0);

    params_.assign((input_size_ + output_size_) * output_size_ * cnt_layers_, 0);
    grad_.assign(params_.size(), 0);

    for (size_t i = 0; i < sequence_length_ * cnt_layers_; ++i) {
        auto activation = std::unique_ptr<IActivation>(activation_factory_());
        cells_.emplace_back(Linear(input_size + output_size, output_size, std::unique_ptr<IBlas>(blas_factory_())),
                            std::move(activation));
    }
}
