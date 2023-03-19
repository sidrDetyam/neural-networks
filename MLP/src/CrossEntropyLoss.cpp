//
// Created by sidr on 19.03.23.
//
#include "../include/CrossEntropyLoss.h"
#include "../include/Utils.h"
#include <cmath>

std::pair<double, Batch> CrossEntropyLoss::apply(const Batch &batch, const std::vector<int> &one_hot) {

    ASSERT(one_hot.size() == batch.getBsize());
    Batch softMaxBatch = softMax(batch);

    double loss = 0.;
    for (size_t i=0; i<batch.getBsize(); ++i){
        loss -= log(softMaxBatch[i][one_hot[i]]);
        softMaxBatch[i][one_hot[i]] -= 1.;
    }

    return {loss, softMaxBatch};
}

Batch CrossEntropyLoss::softMax(const Batch &batch) {

    Batch softMaxBatch(batch.getBsize(), batch.getFeatureSize());

    for (size_t i = 0; i < batch.getBsize(); ++i) {
        double exp_sum = 0.;
        for (size_t j = 0; j < batch.getFeatureSize(); ++j) {
            exp_sum += exp(batch[i][j]);
        }

        for (size_t j = 0; j < batch.getFeatureSize(); ++j) {
            softMaxBatch[i][j] = exp(batch[i][j]) / exp_sum;
        }
    }

    return softMaxBatch;
}

