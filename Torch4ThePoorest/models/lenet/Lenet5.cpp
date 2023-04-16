//
// Created by sidr on 16.04.23.
//

#include "Lenet5.h"
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
#include "Dropout.h"
#include "Tanh.h"
#include "WeightInitializers.h"
#include "TransformDataLoaderDecorator.h"

nn::Linear *nn::linearLayerCreator(const size_t input,
                               const size_t output) {
    return new nn::Linear(input, output,
                          nn::random_vector_gauss(input * output, 0, 0.3),
                          nn::random_vector_gauss(output, 0, 0.1),
                          std::make_unique<CpuBlas>());
}

nn::Conv2d *nn::conv2DCreator(const size_t in_channels, const size_t out_channels) {
    return new nn::Conv2d(in_channels, out_channels, 5,
                          CpuBlas::of(),
                          nn::random_vector_gauss(in_channels * out_channels * 5 * 5 + out_channels, 0,
                                                  0.1),
                          true);
}


nn::Sequential nn::lenet5_model() {
    std::vector<std::unique_ptr<nn::ILayer>> layers;

    layers.emplace_back(new nn::Reshaper({784}, {1, 28, 28}));

    layers.emplace_back(conv2DCreator(1, 6)); // 1,28,28 -> 6,24,24
    layers.emplace_back(new nn::Tanh());
    layers.emplace_back(new nn::AvgPolling(2, 2, 6)); //6,24,24 -> 6,12,12

    layers.emplace_back(conv2DCreator(6, 16)); //6,12,12 -> 16, 8, 8
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

nn::Sequential nn::l_model() {
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
