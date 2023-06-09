//
// Created by sidr on 19.03.23.
//
#include "CrossEntropyLoss.h"
#include "Utils.h"
#include <cmath>

using namespace nn;

std::pair<double, Tensor> CrossEntropyLoss::apply(const Tensor &batch, const std::vector<int> &one_hot) const {

    ASSERT_RE(one_hot.size() == batch.getBsize());
    Tensor softMaxBatch = softMax(batch);

    double loss = 0.;
    for (size_t i=0; i<batch.getBsize(); ++i){
        loss -= log(softMaxBatch[i][one_hot[i]]);
        softMaxBatch[i][one_hot[i]] -= 1.;
    }

    return {loss, std::move(softMaxBatch)};
}

Tensor CrossEntropyLoss::softMax(const Tensor &batch) {

    Tensor softMaxBatch = batch;
    for(int i=0; i<batch.getBsize(); ++i){
        double mx = *std::max_element(batch[i], batch[i+1]);
        for(int j=0; j<batch.getFeatureSize(); ++j){
            softMaxBatch[i][j] -= mx;
        }
    }

    for (size_t i = 0; i < batch.getBsize(); ++i) {
        double exp_sum = 0.;
        for (size_t j = 0; j < batch.getFeatureSize(); ++j) {
            exp_sum += exp(softMaxBatch[i][j]);
        }

        for (size_t j = 0; j < batch.getFeatureSize(); ++j) {
            softMaxBatch[i][j] = exp(softMaxBatch[i][j]) / exp_sum;
            //softMaxBatch[i][j] -= batch[i][j] * 1e-12;
        }
    }

    return softMaxBatch;
}
