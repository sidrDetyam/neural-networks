#include "SgdOptimizerCreator.h"

//
// Created by sidr on 26.03.23.
//
IOptimizer* SgdOptimizerCreator::create(ILayer *layer) {
    return new SgdOptimizer(layer, m_coff_, step_coff_, blas_);
}

SgdOptimizerCreator::SgdOptimizerCreator(double m_coff, double step_coff, std::shared_ptr<IBlas> blas):
    m_coff_(m_coff),
    step_coff_(step_coff),
    blas_(std::move(blas)){

}
