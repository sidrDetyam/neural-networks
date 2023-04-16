//
// Created by sidr on 15.04.23.
//

#ifndef TORCH4THEPOOREST_CLASSIFICATION_H
#define TORCH4THEPOOREST_CLASSIFICATION_H

#include "IDataLoader.h"
#include "Sequential.h"
#include "IClassificationLossFunction.h"

namespace nn{

    std::vector<int> to_one_hot(const nn::Tensor &tensor);

    std::pair<double, Tensor> classification_test(Sequential& model,
                                                  int cnt_of_classes,
                                                  IDataLoader& data_loader,
                                                  const IClassificationLostFunction& loss,
                                                  bool show_progress=false);

    double accuracy(const Tensor& table);
}

#endif //TORCH4THEPOOREST_CLASSIFICATION_H
