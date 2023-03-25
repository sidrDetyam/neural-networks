//
// Created by sidr on 24.03.23.
//
#include "Model.h"
#include <utility>
#include "CpuBlas.h"


Model::Model(std::vector<std::unique_ptr<ILayer>> layers): layers_(std::move(layers)) {

    for(auto & layer : layers_){
        optimizers_.emplace_back(layer.get(), 0.9, 0.1, std::make_unique<CpuBlas>());
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
    for(auto & layer : layers_){
        b = layer->backward(b);
    }
}

void Model::step() {
    for(auto & opt : optimizers_){
        opt.step();
    }
}

