//
// Created by sidr on 24.03.23.
//
#include "Model.h"

#include <utility>

Model::Model(std::vector<std::unique_ptr<ILayer>> layers): layers_(std::move(layers)) {

}

