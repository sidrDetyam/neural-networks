//
// Created by sidr on 26.03.23.
//

#ifndef MLP_SGDOPTIMIZERCREATOR_H
#define MLP_SGDOPTIMIZERCREATOR_H

#include "IOptimizerCreator.h"
#include "SgdOptimizer.h"

class SgdOptimizerCreator: public IOptimizerCreator{
public:
    explicit SgdOptimizerCreator(double m_coff, double lr, std::shared_ptr<IBlas> blas);

    IOptimizer* create(ILayer *layer) override;

private:
    double m_coff_;
    double lr_;
    std::shared_ptr<IBlas> blas_;
};

#endif //MLP_SGDOPTIMIZERCREATOR_H
