//
// Created by sidr on 24.03.23.
//

#ifndef MLP_SGDOPTIMIZER_H
#define MLP_SGDOPTIMIZER_H

#include <ILayer.h>
#include "IBlas.h"
#include "IOptimizer.h"

class SgdOptimizer: public IOptimizer{
public:
    explicit SgdOptimizer(ILayer* layer, double m_coff_, double lr, std::shared_ptr<IBlas> blas);

    void step() override;

private:
    ILayer* layer_;
    std::vector<double> m_;
    bool isFirst_ = true;
    double m_coff_;
    double lr_;
    std::shared_ptr<IBlas> blas_;
};

#endif //MLP_SGDOPTIMIZER_H
