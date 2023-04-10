//
// Created by sidr on 10.04.23.
//

#include "CpuBlas.h"
#include "Tensor.h"
#include <vector>
#include <iostream>
#include "Conv2d.h"
#include "CsvDataLoader.h"
#include "Sequential.h"
#include "Linear.h"
#include "Utils.h"
#include "ReLU.h"
#include "AvgPolling.h"


nn::Linear *linearLayerCreator(const size_t input,
                               const size_t output) {
    return new nn::Linear(input, output,
                          random_vector_gauss(input * output, 0, 1),
                          random_vector_gauss(output, 0, 1),
                          std::make_unique<CpuBlas>());
}

nn::Conv2d *conv2DCreator(const size_t in_channels,
                          const size_t out_channels,
                          const size_t kernel) {
    return new nn::Conv2d(in_channels, out_channels, kernel, kernel,
                          CpuBlas::of(),
                          random_vector_gauss(in_channels*out_channels*kernel*kernel, 0, 1));
}


nn::Sequential lenet5_model() {
    std::vector<std::unique_ptr<nn::ILayer>> layers;

    layers.emplace_back(conv2DCreator(1, 6, 5)); // 1,28,28 -> 6,24,24
    layers.emplace_back(new nn::ReLU(CpuBlas::of()));
    layers.emplace_back(new nn::AvgPolling(2, 2, 6)); //6,24,24 -> 6,12,12

    layers.emplace_back(conv2DCreator(6, 16, 5)); //6,12,12 -> 16, 8, 8
    layers.emplace_back(new nn::ReLU(CpuBlas::of()));
    layers.emplace_back(new nn::AvgPolling(2, 2, 16)); //16,8,8 -> 16,4,4
    layers.emplace_back(conv2DCreator(16, 120, 4));

    layers.emplace_back(linearLayerCreator(120, ))


    layers.emplace_back(linearLayerCreator(22, 11));
    layers.emplace_back(new ReLU(CpuBlas::of()));
    layers.emplace_back(linearLayerCreator(11, 2));

    SgdOptimizerCreator sgd_creator(0.95, 0.05, std::make_shared<CpuBlas>());

    Sequential model(std::move(layers), std::make_unique<SgdOptimizerCreator>(std::move(sgd_creator)));

    return model;
}


int main() {
    nn::CsvDataLoader loader(2, true,
                             "/home/sidr/PycharmProjects/neural-networks/Torch4ThePoorest/data/mnist_test.csv",
                             785, {0});

    auto b = loader.next_batch();
    cout << b.second << endl << endl;

    cout << b.first << endl;
}