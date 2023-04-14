#include "Tensor.h"
#include "Dropout.h"

//
// Created by sidr on 11.04.23.
//
using namespace nn;

Tensor nn::DropoutLayer::forward(Tensor &&input) {
    m_input = std::move(input);
    m_mask.resize(m_input.size());
    std::bernoulli_distribution dist(1.0 - m_dropoutProbability);
    for (size_t i = 0; i < m_mask.size(); ++i) {
        m_mask[i] = dist(m_rng);
        m_input.data()[i] *= m_mask[i];
    }
    return m_input;
}

Tensor nn::DropoutLayer::backward(const Tensor &output) {
    Tensor gradient = output;
    ASSERT_RE(output.size() == m_mask.size());
    for (size_t i = 0; i < gradient.size(); ++i) {
        gradient.data()[i] *= m_mask[i];
    }
    return gradient;
}

std::vector<double> &nn::DropoutLayer::getParametersGradient() {
    return m_emptyGradient;
}

std::vector<double> &nn::DropoutLayer::getParameters() {
    return m_emptyGradient;
}

[[maybe_unused]] DropoutLayer::DropoutLayer(double dropoutProbability): m_dropoutProbability(dropoutProbability) {

}
