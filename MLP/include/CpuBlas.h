//
// Created by sidr on 18.03.23.
//

#ifndef MLP_CPUBLAS_H
#define MLP_CPUBLAS_H

#include "IBlas.h"
#include <cblas.h>
#include <memory>

class CpuBlas: public IBlas{
public:
    void dgemm(const double *a, const double *b, bool isATransposed, bool isBTransposed, double *c, int m, int n, int k, double beta) override;

    void col_sum(const double *a, double *res, int m, int n, double beta) override;

    static std::unique_ptr<CpuBlas> of();

    void scale(double *a, int n, double scale) override;

    void daxpby(int n, double *a, double alpha, double *b, double beta) override;

    void debug(int n, double *a, double alpha, double *b, double beta);
};


#endif //MLP_CPUBLAS_H
