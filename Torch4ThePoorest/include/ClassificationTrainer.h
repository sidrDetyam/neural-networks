//
// Created by sidr on 15.04.23.
//

#ifndef TORCH4THEPOOREST_CLASSIFICATIONTRAINER_H
#define TORCH4THEPOOREST_CLASSIFICATIONTRAINER_H

#include "IDataLoader.h"
#include "Sequential.h"
#include "IClassificationLossFunction.h"

namespace nn{

    void classification_train(Sequential& model, IDataLoader& data_loader, const IClassificationLostFunction& loss);

}

#endif //TORCH4THEPOOREST_CLASSIFICATIONTRAINER_H
