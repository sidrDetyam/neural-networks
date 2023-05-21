//
// Created by sidr on 21.05.23.
//

#include <iostream>
#include "Linear.h"
#include "WeightInitializers.h"
#include "CpuBlas.h"
#include "Sequential.h"
#include "Reshaper.h"
#include "SgdOptimizerCreator.h"
#include "Rnn.h"
#include "Tanh.h"
#include "ReLU.h"
#include "CsvDataLoader.h"
#include "CachingDataLoader.h"
#include "MSELoss.h"
#include "Tqdm.h"

using namespace nn;
using namespace std;

nn::Linear *linearLayerCreator(const size_t input,
                               const size_t output) {
    return new nn::Linear(input, output,
                          nn::random_vector_gauss(input * output, 0, 0.4),
                          nn::random_vector_gauss(output, 0, 0.4),
                          std::make_unique<CpuBlas>());
}

const size_t seq_len = 8;
const size_t input_size = 17;

nn::Sequential rnn_model() {
    std::vector<std::unique_ptr<nn::ILayer>> layers;
    const size_t hidden_size = 8;
    const size_t output_size = 1;
    const size_t cnt_layers = 3;

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

    nn::SgdOptimizerCreator sgd_creator(0.97, 0.00001, std::make_shared<CpuBlas>());
    nn::Sequential model(std::move(layers), std::make_unique<nn::SgdOptimizerCreator>(std::move(sgd_creator)));

    return model;
}


int main() {

    string train_path = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/data/train_lab3.csv";
//    string test_path = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/Torch4ThePoorest/data/mnist_test.csv";
//    string model_dump_path = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/Torch4ThePoorest/data/lenet5_model";
    const int epochs = 5000;

    nn::CsvDataLoader loader_train(64, true,
                                   train_path,
                                   seq_len*input_size+1, {0});

//    nn::CsvDataLoader loader_test(130, true,
//                                  test_path,
//                                  785, {0});

    //nn::Sequential lenet = lenet5_model();
    nn::Sequential lenet = rnn_model();

    CachingDataLoader train(loader_train);
//    CachingDataLoader test(TransformDataLoaderDecorator(loader_test, normalization));

    cout << "Loaded\n\n";

    nn::MSELoss loss;

    Tensor out_;
    Tensor correct_;
    for (int e = 0; e < epochs; ++e) {
//        train.shuffle();
        train.reset();
        double err = 0;
        int total = 0;
        cout << "Epoch " << e+1 << endl;

        Tqdm tqdm(3);
        for (int bi = tqdm.start((int)train.size()); !tqdm.is_end();) {
            const auto batch = train.next_batch();
            Tensor input = batch.first;

            Tensor out = lenet.forward(std::move(input));
            if(out.get_shape()[0] > 60){
                out_ = out;
                correct_ = batch.second;
            }
            auto l = loss.apply(out, batch.second);
            lenet.backward(l.second);
            lenet.step();

            total += 1;//(int) out.getBsize();
            err += l.first;

            bi = tqdm.next();

//            if(tqdm.is_end()){
////                cout<< out.get_shape().size() << std::endl;
//            }
        }

        cout << "Loss: " << err / total << endl;
        if(err / total < 320){
            break;
        }

//        sleep(5);
    }

    std::cout << out_.size() << endl;
    std::cout << correct_.size() << endl;

    return 0;
}