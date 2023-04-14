//
// Created by sidr on 18.03.23.
//

#ifndef MLP_CPUBLAS_H
#define MLP_CPUBLAS_H

#include "IBlas.h"
#include <cblas.h>
#include <memory>

namespace details{
    enum CBLAS_ORDER to_cblas_order(MatrixOrder order);

    enum CBLAS_TRANSPOSE to_cblas_transpose(Transpose trans);
}

class CpuBlas: public IBlas{
public:
    void dgemm(const double *a, const double *b, bool isATransposed, bool isBTransposed, double *c, int m, int n, int k, double beta) override;

    void col_sum(const double *a, double *res, int m, int n, double beta) override;

    static std::unique_ptr<CpuBlas> of();

    void scale(double *a, int n, double scale) override;

    void daxpby_full(int n, const double *a, double alpha, int inca, double *b, double beta, int incb) override;

    void daxpby(int n, const double *a, double alpha, double *b, double beta) override;

    void dgemm_full(MatrixOrder order, Transpose trans_a, Transpose trans_b, int m, int n, int k, double alpha,
                    const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc) override;

    void element_wise_mult(int n, const double *a, const double *b, double *c) override;

    void convolve(const double *A, const double *W, double *C, int N, int M, int R, int S, double beta) override;
};


#endif //MLP_CPUBLAS_H
