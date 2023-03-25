//
// Created by sidr on 25.03.23.
//
#include "Optimizer.h"
#include "Utils.h"
#include "CpuBlas.h"

Optimizer::Optimizer(ILayer *layer, double m_coff, double step_coff, std::unique_ptr<IBlas>&& blas):
    layer_(layer),
    m_coff_(m_coff),
    blas_(std::move(blas)),
    step_coff_(step_coff){

}

void Optimizer::step() {

    std::vector<double>& grad = layer_->getParametersGradient();
    std::vector<double>& params = layer_->getParameters();
    ASSERT(grad.size() == layer_->getParameters().size());

    auto b = reinterpret_cast<CpuBlas*>(blas_.get());

    if(isFirst_){
        isFirst_ = false;
        m_ = grad;
    }
    else{
        ASSERT(grad.size() == m_.size());
        b->debug((int) grad.size(), grad.data(), (1 - m_coff_), m_.data(), m_coff_);
    }

    auto mdata = m_.data();
    auto params_d = params.data();
    b->debug((int)grad.size(), mdata, -step_coff_, params_d, 1.);
}
