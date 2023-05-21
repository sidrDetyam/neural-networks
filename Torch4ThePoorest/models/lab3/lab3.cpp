//
// Created by sidr on 21.05.23.
//

#include "lab3.h"

using namespace nn;
using namespace std;

nn::Linear *linearLayerCreator(const size_t input,
                               const size_t output) {
    return new nn::Linear(input, output,
                          nn::random_vector_gauss(input * output, 0, 0.4),
                          nn::random_vector_gauss(output, 0, 0.4),
                          std::make_unique<CpuBlas>());
}

nn::Sequential rnn_model() {
    std::vector<std::unique_ptr<nn::ILayer>> layers;
    const size_t hidden_size = 12;
    const size_t output_size = 1;
    const size_t cnt_layers = 2;

    layers.emplace_back(new nn::Reshaper({input_size * seq_len}, {seq_len, input_size}));

    std::function<nn::ILayer *()> activation_factory = []() {
        auto ptr = new nn::Tanh();
//        cout << "af " << ptr << endl;
        return ptr;
    };
    std::function<IBlas *()> blas_factory = []() {
        return new CpuBlas();
    };

    layers.emplace_back(new nn::Rnn(input_size, hidden_size, cnt_layers, seq_len,
                                    std::move(activation_factory), std::move(blas_factory)));

    nn::random_vector_gauss(layers.back()->getParameters(), 0., 0.05);

//    layers.emplace_back(new nn::Linear(hidden_size, hidden_size/2, CpuBlas::of()));
    layers.emplace_back(linearLayerCreator(hidden_size, hidden_size/2));
    layers.emplace_back(new nn::ReLU(CpuBlas::of()));
    layers.emplace_back(linearLayerCreator(hidden_size/2, output_size));
//    layers.emplace_back(new nn::ReLU(CpuBlas::of()));
//    layers.emplace_back(linearLayerCreator(hidden_size/4, output_size));

    nn::SgdOptimizerCreator sgd_creator(0.97, 0.00001, std::make_shared<CpuBlas>());
    nn::Sequential model(std::move(layers), std::make_unique<nn::SgdOptimizerCreator>(std::move(sgd_creator)));

    return model;
}
