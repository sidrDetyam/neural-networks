//
// Created by sidr on 18.03.23.
//
#include <memory>

#include "CpuBlas.h"
#include <vector>

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

void CpuBlas::daxpby(int n, const double *a, double alpha, double *b, double beta) {
    if (n != 0) {
        cblas_daxpby(n, alpha, a, 1, beta, b, 1);
    }
}

void CpuBlas::dgemm_full(MatrixOrder order, Transpose trans_a, Transpose trans_b, int m, int n, int k, double alpha,
                         const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc) {

    cblas_dgemm(details::to_cblas_order(order),
                details::to_cblas_transpose(trans_a),
                details::to_cblas_transpose(trans_b),
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CpuBlas::element_wise_mult(int n, const double *a, const double *b, double *c) {
    cblas_dsbmv(CblasRowMajor, CblasLower, n, 0, 1.0, a, 1, b, 1, 0.0, c, 1);
}

void CpuBlas::convolve(const double *const A,
                       const double *const W,
                       double *const C,
                       const int N, const int M, const int K1, const int K2,
                       const double beta) {
    const int P = N - K1 + 1;
    const int Q = M - K2 + 1;

    std::vector<double> tmp(K1 * K2);

    for (int i = 0; i < P; i++) {
        for (int j = 0; j < Q; j++) {

            for (int k1 = 0; k1 < K1; ++k1) {
                for (int k2 = 0; k2 < K2; ++k2) {
                    tmp[k2 + k1 * K2] = A[M * (i + k1) + j + k2];
                }
            }

            C[i * Q + j] = C[i * Q + j] * beta + cblas_ddot(K1 * K2, tmp.data(), 1, W, 1);
        }
    }
}

enum CBLAS_ORDER details::to_cblas_order(MatrixOrder order) {
    return order == ROW_ORDER ? CblasRowMajor : CblasColMajor;
}

enum CBLAS_TRANSPOSE details::to_cblas_transpose(Transpose trans) {
    return trans == TRANS ? CblasTrans : CblasNoTrans;
}
