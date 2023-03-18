//
// Created by sidr on 18.03.23.
//
#include "../include/CpuBlas.h"

void CpuBlas::dgemm(const double *a, const double *b, double *c, int m, int n, int k) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1., a, k, b, n, 1., c, n);
}
