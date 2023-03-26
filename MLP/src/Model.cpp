//
// Created by sidr on 24.03.23.
//
#include "Model.h"
#include <ranges>
#include <utility>
#include "CpuBlas.h"


Model::Model(std::vector<std::unique_ptr<ILayer>> layers,
             std::unique_ptr<IOptimizerCreator> &&creator):
    layers_(std::move(layers)){

    for(auto & layer : layers_){
        optimizers_.emplace_back(creator->create(layer.get()));
    }
}

Batch Model::forward(Batch &&batch) {

    Batch out = std::move(batch);
    for(auto & layer : layers_){
        out = layer->forward(std::move(out));
    }

    return out;
}

void Model::backward(const Batch &output) {

    Batch b = output;
    for(auto & layer : std::ranges::reverse_view(layers_)){
        b = layer->backward(b);
    }
}

void Model::step() {
    for(auto & opt : optimizers_){
        opt->step();
    }
}

