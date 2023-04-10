//
// Created by sidr on 24.03.23.
//
#include "Sequential.h"
#include <ranges>
#include <utility>
#include "CpuBlas.h"

using namespace nn;

Sequential::Sequential(std::vector<std::unique_ptr<ILayer>> layers,
                       std::unique_ptr<IOptimizerCreator> &&creator):
    layers_(std::move(layers)){

    for(auto & layer : layers_){
        optimizers_.emplace_back(creator->create(layer.get()));
    }
}

Tensor Sequential::forward(Tensor &&batch) {

    Tensor out = std::move(batch);
    for(auto & layer : layers_){
        out = layer->forward(std::move(out));
    }

    return out;
}

void Sequential::backward(const Tensor &output) {

    Tensor b = output;
    for(auto & layer : std::ranges::reverse_view(layers_)){
        b = layer->backward(b);
    }
}

void Sequential::step() {
    for(auto & opt : optimizers_){
        opt->step();
    }
}

