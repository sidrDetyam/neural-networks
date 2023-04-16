//
// Created by sidr on 10.04.23.
//

#include "CpuBlas.h"
#include <vector>
#include <iostream>
#include "Conv2d.h"
#include "CsvDataLoader.h"
#include "Sequential.h"
#include "Linear.h"
#include "AvgPolling.h"
#include "Reshaper.h"
#include "SgdOptimizerCreator.h"
#include "CrossEntropyLoss.h"
#include "Dropout.h"
#include "Tanh.h"
#include "Tqdm.h"
#include "WeightInitializers.h"
#include "TransformDataLoaderDecorator.h"
#include "CachingDataLoader.h"
#include "Classification.h"

namespace {

    nn::Linear *linearLayerCreator(const size_t input,
                                   const size_t output) {
        return new nn::Linear(input, output,
                              nn::random_vector_gauss(input * output, 0, 0.3),
                              nn::random_vector_gauss(output, 0, 0.1),
                              std::make_unique<CpuBlas>());
    }

    nn::Conv2d *conv2DCreator(const size_t in_channels,
                              const size_t out_channels,
                              const size_t kernel) {
        return new nn::Conv2d(in_channels, out_channels, kernel,
                              CpuBlas::of(),
                              nn::random_vector_gauss(in_channels * out_channels * kernel * kernel + out_channels, 0,
                                                      0.1),
                              true);
    }


    nn::Sequential lenet5_model() {
        std::vector<std::unique_ptr<nn::ILayer>> layers;

        layers.emplace_back(new nn::Reshaper({784}, {1, 28, 28}));

        layers.emplace_back(conv2DCreator(1, 6, 5)); // 1,28,28 -> 6,24,24
        layers.emplace_back(new nn::Tanh());
        layers.emplace_back(new nn::AvgPolling(2, 2, 6)); //6,24,24 -> 6,12,12

        layers.emplace_back(conv2DCreator(6, 16, 5)); //6,12,12 -> 16, 8, 8
        layers.emplace_back(new nn::Tanh());
        layers.emplace_back(new nn::AvgPolling(2, 2, 16)); //16,8,8 -> 16,4,4

        layers.emplace_back(new nn::Reshaper({16, 4, 4}, {256}));

        layers.emplace_back(linearLayerCreator(256, 120));
        layers.emplace_back(new nn::Tanh());
        layers.emplace_back(linearLayerCreator(120, 84));
        layers.emplace_back(new nn::Tanh());
        layers.emplace_back(linearLayerCreator(84, 10));

        nn::SgdOptimizerCreator sgd_creator(0.9, 0.01, std::make_shared<CpuBlas>());
        nn::Sequential model(std::move(layers), std::make_unique<nn::SgdOptimizerCreator>(std::move(sgd_creator)));

        return model;
    }

    nn::Sequential l_model() {
        std::vector<std::unique_ptr<nn::ILayer>> layers;

        layers.emplace_back(linearLayerCreator(784, 129));
        layers.emplace_back(new nn::Tanh());
        //layers.emplace_back(new nn::ReLU(CpuBlas::of()));
//        layers.emplace_back(new nn::DropoutLayer(0.3));
        layers.emplace_back(linearLayerCreator(129, 84));
//    layers.emplace_back(new nn::ReLU(CpuBlas::of()));
        layers.emplace_back(new nn::Tanh());
//        layers.emplace_back(new nn::DropoutLayer(0.3));
        layers.emplace_back(linearLayerCreator(84, 30));
//    layers.emplace_back(new nn::ReLU(CpuBlas::of()));
        layers.emplace_back(new nn::Tanh());
//        layers.emplace_back(new nn::DropoutLayer(0.3));
        layers.emplace_back(linearLayerCreator(30, 10));

        nn::SgdOptimizerCreator sgd_creator(0.9, 0.01, std::make_shared<CpuBlas>());
        nn::Sequential model(std::move(layers), std::make_unique<nn::SgdOptimizerCreator>(std::move(sgd_creator)));

        return model;
    }
}

int main() {
    using namespace nn;
    using namespace std;

    nn::CsvDataLoader loader_train(130, true,
                                   "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/Torch4ThePoorest/data/mnist_train.csv",
                                   785, {0});

    nn::CsvDataLoader loader_test(130, true,
                                  "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/Torch4ThePoorest/data/mnist_test.csv",
                                  785, {0});

    nn::Sequential lenet = lenet5_model();
    //nn::Sequential lenet = l_model();


    auto normalization = [](batch_t &b){
        for (auto &i: b.first.data()) {
            i /= 255.;
        }
    };

    CachingDataLoader train(TransformDataLoaderDecorator(loader_train, normalization));
    CachingDataLoader test(TransformDataLoaderDecorator(loader_test, normalization));

    cout << "Loaded\n\n";

    nn::CrossEntropyLoss loss;

    for (int e = 0; e < 1000; ++e) {
        train.shuffle();

        double err = 0;
        int total = 0;

        cout << "Epoch " << e+1 << endl;

        Tqdm tqdm(3);
        for (int bi = tqdm.start((int)train.size()); !tqdm.is_end();) {
            int correct = 0;
            const auto batch = train.next_batch();

            nn::Tensor b = batch.first;
            auto out = lenet.forward(std::move(b));
            total += (int) out.getBsize();
            std::vector<int> one_hot;
            for (auto i: batch.second.data()) {
                one_hot.push_back((int) i);
            }
            auto l = loss.apply(out, one_hot);
            lenet.backward(l.second);
            lenet.step();

            err += l.first;

            for (size_t i = 0; i < out.getBsize(); ++i) {
                long cl = std::max_element(out[i], out[i + 1]) - out[i];
                correct += cl == one_hot[i];
            }

            bi = tqdm.next();
            tqdm << "  Training... " << bi << "/" << train.size()
                 << " Mean loss(epoch): " << err / total
                 << ", Mean loss(batch): " << l.first / (double) l.second.getBsize()
                 << ", Accuracy(batch): " << (double) correct / (double) l.second.getBsize();
        }

        test.reset();
        cout << endl;
        auto test_result = classification_test(lenet, 10, test, loss);
        cout << "  Testing result: " << err / total << " " << test_result.first << " " << accuracy(test_result.second) << endl;
        sleep(5);
    }
}
