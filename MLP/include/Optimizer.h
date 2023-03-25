//
// Created by sidr on 24.03.23.
//

#ifndef MLP_OPTIMIZER_H
#define MLP_OPTIMIZER_H

#include <ILayer.h>
#include "IBlas.h"

class Optimizer{
public:
    explicit Optimizer(ILayer* layer, double m_coff_, double step_coff_, std::unique_ptr<IBlas>&& blas);

    void step();


private:
    ILayer* layer_;
    std::vector<double> m_;
    bool isFirst_ = true;
    double m_coff_;
    double step_coff_;
    std::unique_ptr<IBlas>&& blas_;
};

#endif //MLP_OPTIMIZER_H
