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

    virtual void daxpby(int n, double *a, double alpha, double *b, double beta) = 0;

    [[maybe_unused]] virtual void dgemm_full(MatrixOrder order, Transpose trans_a, Transpose trans_b,
                                             int m, int n, int k,
                                             double alpha, const double *a, int lda,
                                             const double *b, int ldb, double beta,
                                             double *c, int ldc) = 0;

    virtual ~IBlas() = default;
};

#endif //MLP_IBLAS_H
