//
// Created by sidr on 18.03.23.
//

#ifndef MLP_CPUBLAS_H
#define MLP_CPUBLAS_H

#include "IBlas.h"
#include <cblas.h>

class CpuBlas: public IBlas{
public:
    void dgemm(const double *a, const double *b, double *c, int m, int n, int k) override;
};


#endif //MLP_CPUBLAS_H
