//
// Created by sidr on 18.03.23.
//
#include <memory>
#include <iostream>

#include "CpuBlas.h"

void
CpuBlas::dgemm(const double *a, const double *b, bool isATransposed, bool isBTransposed, double *c, int m, int n, int k,
               double beta) {
    cblas_dgemm(CblasRowMajor,
                isATransposed ? CblasTrans : CblasNoTrans,
                isBTransposed ? CblasTrans : CblasNoTrans,
                m, n, k, 1., a,
            //TODO bruh
                isATransposed ? m : k,
                b,
                isBTransposed ? k : n,
                beta, c, n);
}

std::unique_ptr<CpuBlas> CpuBlas::of() {
    return std::make_unique<CpuBlas>();
}

void CpuBlas::col_sum(const double *a, double *res, int m, int n, double beta) {
    for (int i = 0; i < n; ++i) {
        res[i] *= beta;
        for (int j = 0; j < m; ++j) {
            res[i] += a[j * n + i];
        }
    }
}

void CpuBlas::scale(double *a, int n, double scale) {
    cblas_dscal(n, scale, a, 1);
}

void CpuBlas::daxpby(int n, double *a, double alpha, double *b, double beta) {
    if(n==0) {
        std::cout << "daxpby" << std::endl;
        cblas_daxpby(n, alpha, a, 1, beta, b, 1);
    }
}

void CpuBlas::debug(int n, double *a, double alpha, double *b, double beta) {
    if(n!=0) {
        //std::cout << "debug" << std::endl;
        cblas_daxpby(n, alpha, a, 1, beta, b, 1);
    }
}
