//
// Created by sidr on 18.03.23.
//

#ifndef MLP_IBLAS_H
#define MLP_IBLAS_H

enum MatrixOrder {
    ROW_ORDER,
    COLUMN_ORDER [[maybe_unused]]
};


enum Transpose {
    TRANS,
    NO_TRANS [[maybe_unused]]
};


class IBlas {
public:
    virtual void dgemm(const double *a, const double *b, bool isATransposed, bool isBTransposed,
                       double *c, int m, int n, int k, double beta) = 0;

    virtual void col_sum(const double *a, double *res, int m, int n, double beta) = 0;

    virtual void scale(double *a, int n, double scale) = 0;

    virtual void daxpby(int n, const double *a, double alpha, double *b, double beta) = 0;

    virtual void daxpby_full(int n, const double *a, double alpha, int inca, double *b, double beta, int incb) = 0;

    virtual void element_wise_mult(int n, const double *a, const double *b, double *c) = 0;

    [[maybe_unused]] virtual void dgemm_full(MatrixOrder order, Transpose trans_a, Transpose trans_b,
                                             int m, int n, int k,
                                             double alpha, const double *a, int lda,
                                             const double *b, int ldb, double beta,
                                             double *c, int ldc) = 0;

    virtual void convolve(const double* A, const double* W, double* C, int N, int M, int R, int S, double beta) = 0;

    virtual ~IBlas() = default;
};

#endif //MLP_IBLAS_H
