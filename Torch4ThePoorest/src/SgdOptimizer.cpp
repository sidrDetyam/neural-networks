//
// Created by sidr on 25.03.23.
//
#include "SgdOptimizer.h"

#include <utility>
#include "Utils.h"
#include "CpuBlas.h"

using namespace nn;

SgdOptimizer::SgdOptimizer(ILayer *layer, double m_coff, double lr, std::shared_ptr<IBlas> blas):
        layer_(layer),
        m_coff_(m_coff),
        blas_(std::move(blas)),
        lr_(lr){

}

void SgdOptimizer::step() {

    std::vector<double>& grad = layer_->getParametersGradient();
    std::vector<double>& params = layer_->getParameters();
    ASSERT_RE(grad.size() == layer_->getParameters().size());

    if(isFirst_){
        isFirst_ = false;
        m_ = grad;
        //blas_->scale(m_.data(), (int)m_.size(), -lr_);
    }
    else{
        ASSERT_RE(grad.size() == m_.size());
        blas_->daxpby((int) grad.size(), grad.data(), (1 - m_coff_), m_.data(), m_coff_);
    }

    blas_->daxpby((int)grad.size(), m_.data(), -lr_, params.data(), 1.);
}
