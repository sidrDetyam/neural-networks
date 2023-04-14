#include "SgdOptimizerCreator.h"

//
// Created by sidr on 26.03.23.
//

using namespace nn;

IOptimizer* SgdOptimizerCreator::create(ILayer *layer) {
    return new SgdOptimizer(layer, m_coff_, lr_, blas_);
}

SgdOptimizerCreator::SgdOptimizerCreator(double m_coff, double lr, std::shared_ptr<IBlas> blas):
        m_coff_(m_coff),
        lr_(lr),
        blas_(std::move(blas)){

}
