//
// Created by sidr on 24.03.23.
//
#include "Sequential.h"
#include <ranges>
#include <utility>
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "CpuBlas.h"
#include <boost/serialization/vector.hpp>

using namespace nn;

Sequential::Sequential(std::vector<std::unique_ptr<ILayer>> layers,
                       std::unique_ptr<IOptimizerCreator> &&creator) :
        layers_(std::move(layers)) {

    for (auto &layer: layers_) {
        optimizers_.emplace_back(creator->create(layer.get()));
    }
}

Tensor Sequential::forward(Tensor &&batch) {

    Tensor out = std::move(batch);
    for (auto &layer: layers_) {
        out = layer->forward(std::move(out));
    }

    return out;
}

Tensor Sequential::backward(const Tensor &output) {

    Tensor b = output;
    for (auto &layer: std::ranges::reverse_view(layers_)) {
        b = layer->backward(b);
    }

    return b;
}

void Sequential::step() {
    for (auto &opt: optimizers_) {
        opt->step();
    }
}

namespace nn {
    std::ofstream &operator<<(std::ofstream &fout, Sequential &model) {
        boost::archive::text_oarchive oa(fout);
        for (auto &layer: model.layers_) {
            oa << *layer;
        }

        return fout;
    }

    std::ifstream &operator>>(std::ifstream &fin, Sequential &model) {
        boost::archive::text_iarchive ia(fin);
        for (auto &layer: model.layers_) {
            ia >> *layer;
        }

        return fin;
    }
}