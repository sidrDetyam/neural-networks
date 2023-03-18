//
// Created by sidr on 18.03.23.
//
#include <memory>

#include "../include/CpuBlas.h"

void CpuBlas::dgemm(const double *a, const double *b, bool isATransposed, bool isBTransposed, double *c, int m, int n, int k, double beta) {
    cblas_dgemm(CblasRowMajor,
                isATransposed? CblasTrans : CblasNoTrans,
                isBTransposed? CblasTrans : CblasNoTrans,
                m, n, k, 1., a, k, b, n, beta, c, n);
}

std::unique_ptr<CpuBlas> CpuBlas::of() {
    return std::make_unique<CpuBlas>();
}
