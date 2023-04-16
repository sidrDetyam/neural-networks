//
// Created by sidr on 15.04.23.
//

#ifndef TORCH4THEPOOREST_CLASSIFICATION_H
#define TORCH4THEPOOREST_CLASSIFICATION_H

#include "IDataLoader.h"
#include "Sequential.h"
#include "IClassificationLossFunction.h"

namespace nn{

    std::pair<double, Tensor> classification_test(Sequential& model,
                                                  int cnt_of_classes,
                                                  IDataLoader& data_loader,
                                                  const IClassificationLostFunction& loss);

    double accuracy(const Tensor& table);
}

#endif //TORCH4THEPOOREST_CLASSIFICATION_H
