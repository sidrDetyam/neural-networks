//
// Created by sidr on 10.04.23.
//

#include "CpuBlas.h"
#include <vector>
#include <iostream>
#include "Conv2dNaive.h"
#include "CsvDataLoader.h"
#include "Sequential.h"
#include "Linear.h"
#include "Utils.h"
#include "ReLU.h"
#include "AvgPolling.h"
#include "Reshaper.h"
#include "SgdOptimizerCreator.h"
#include "CrossEntropyLoss.h"
#include "Dropout.h"
#include "Tanh.h"


nn::Linear *linearLayerCreator(const size_t input,
                               const size_t output) {
    return new nn::Linear(input, output,
                          xavier_init(input * output),
                          xavier_init(output),
                          std::make_unique<CpuBlas>());
}

nn::Conv2dNaive *conv2DCreator(const size_t in_channels,
                               const size_t out_channels,
                               const size_t kernel) {
    return new nn::Conv2dNaive(in_channels, out_channels, kernel, kernel,
                               CpuBlas::of(),
                               xavier_init(in_channels * out_channels * kernel * kernel));
}




nn::Sequential lenet5_model() {
    std::vector<std::unique_ptr<nn::ILayer>> layers;

    layers.emplace_back(new nn::Reshaper({784}, {1, 28, 28}));

    layers.emplace_back(conv2DCreator(1, 6, 5)); // 1,28,28 -> 6,24,24

    //layers.emplace_back(new nn::DropoutLayer(0.3));
    //layers.emplace_back(new nn::ReLU(CpuBlas::of()));
    layers.emplace_back(new nn::Tanh());

    layers.emplace_back(new nn::AvgPolling(2, 2, 6)); //6,24,24 -> 6,12,12

    layers.emplace_back(conv2DCreator(6, 16, 5)); //6,12,12 -> 16, 8, 8

    //layers.emplace_back(new nn::DropoutLayer(0.3));

    //layers.emplace_back(new nn::ReLU(CpuBlas::of()));
    layers.emplace_back(new nn::Tanh());
    layers.emplace_back(new nn::AvgPolling(2, 2, 16)); //16,8,8 -> 16,4,4
    layers.emplace_back(conv2DCreator(16, 120, 4)); //120,1,1

    layers.emplace_back(new nn::Reshaper({120, 1, 1}, {120}));

    layers.emplace_back(linearLayerCreator(120, 84));
    //layers.emplace_back(new nn::ReLU(CpuBlas::of()));
    //layers.emplace_back(new nn::DropoutLayer(0.5));
    layers.emplace_back(new nn::Tanh());

    layers.emplace_back(linearLayerCreator(84, 10));
    //layers.emplace_back(new nn::ReLU(CpuBlas::of()));
    //layers.emplace_back(new nn::DropoutLayer(0.5));
    layers.emplace_back(new nn::Tanh());

    layers.emplace_back(linearLayerCreator(10, 10));

    nn::SgdOptimizerCreator sgd_creator(0.9, 0.01, std::make_shared<CpuBlas>());
    nn::Sequential model(std::move(layers), std::make_unique<nn::SgdOptimizerCreator>(std::move(sgd_creator)));

    return model;
}

nn::Sequential l_model() {
    std::vector<std::unique_ptr<nn::ILayer>> layers;

    layers.emplace_back(linearLayerCreator(784, 129));
    layers.emplace_back(new nn::ReLU(CpuBlas::of()));
    layers.emplace_back(linearLayerCreator(129, 84));
    layers.emplace_back(new nn::ReLU(CpuBlas::of()));
    layers.emplace_back(linearLayerCreator(84, 10));

    nn::SgdOptimizerCreator sgd_creator(0.9, 0.001, std::make_shared<CpuBlas>());
    nn::Sequential model(std::move(layers), std::make_unique<nn::SgdOptimizerCreator>(std::move(sgd_creator)));

    return model;
}

using namespace std;


std::pair<double, double> loss_accuracy(nn::Sequential& model, nn::IClassificationLostFunction& loss, const std::vector<nn::batch_t>& test){
    double err = 0;
    int correct = 0;
    int total = 0;

    int bi = 0;
    for(const auto & batch : test){
        nn::Tensor b = batch.first;
        total += (int)b.getBsize();
        auto out = model.forward(std::move(b));

        std::vector<int> one_hot;
        for(auto i : batch.second.data()){
            one_hot.push_back((int) i);
        }

        auto l = loss.apply(out, one_hot);

        for(size_t i = 0; i < out.getBsize(); ++i){
            long cl = std::max_element(out[i], out[i+1]) - out[i];
            correct += cl == one_hot[i];
        }

        err += l.first;
        cout << "testing " << bi << "/" << test.size() << endl;
        bi++;
        if(bi==20){
            break;
        }
    }

    return {err / total, (double)correct / total};
}


int main() {
    nn::CsvDataLoader loader_train(128, true,
                                   "/home/sidr/PycharmProjects/neural-networks/Torch4ThePoorest/data/mnist_train.csv",
                                   785, {0});

    nn::CsvDataLoader loader_test(128, true,
                                  "/home/sidr/PycharmProjects/neural-networks/Torch4ThePoorest/data/mnist_test.csv",
                                  785, {0});

    std::vector<nn::batch_t> train_batches = loader_train.read_all();
    std::vector<nn::batch_t> test_batches = loader_test.read_all();
    nn::Sequential lenet = lenet5_model();
    //nn::Sequential lenet = l_model();

    for(auto& b : train_batches){
        for(auto & i : b.first.data()){
            i /= 255.;
        }
    }

    for(auto& b : test_batches){
        for(auto & i : b.first.data()){
            i /= 255.;
        }
    }

    cout << "Loaded\n\n";

    nn::CrossEntropyLoss loss;

    for(int e=0; e<1000; ++e){
        double err = 0;
        int total = 0;

        int bi = 0;
        for(const auto & batch : train_batches){
            int correct = 0;

            nn::Tensor b = batch.first;
            auto out = lenet.forward(std::move(b));
            total += (int) out.getBsize();
            std::vector<int> one_hot;
            for(auto i : batch.second.data()){
                one_hot.push_back((int) i);
            }
            auto l = loss.apply(out, one_hot);
            lenet.backward(l.second);
            lenet.step();

            err += l.first;

            for(size_t i = 0; i < out.getBsize(); ++i){
                long cl = std::max_element(out[i], out[i+1]) - out[i];
                correct += cl == one_hot[i];
            }

            cout << "training " << bi << "/" << train_batches.size() << " " << err / total << " "
            << l.first / (double)l.second.getBsize() << " " << (double)correct / l.second.getBsize()<< endl;
            ++bi;

//            if(bi == 40){
//                break;
//            }

//            auto bruh = loss_accuracy(model, loss, test_batches);
//            cout << bruh.first << " " << bruh.second << " " << endl;
//            sleep(1);
        }

        auto val = loss_accuracy(lenet, loss, test_batches);
        cout << e << " " << err / total << " " << val.first << " " << val.second << endl;
        sleep(1);
    }
}
