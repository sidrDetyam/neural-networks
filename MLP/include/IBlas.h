//
// Created by sidr on 18.03.23.
//

#ifndef MLP_IBLAS_H
#define MLP_IBLAS_H

class IBlas{
public:
    virtual void dgemm(const double* a, const double *b, bool isATransposed, bool isBTransposed,
                       double *c, int m, int n, int k, double beta) = 0;

    virtual void col_sum(const double* a, double* res, int m, int n, double beta) = 0;

    virtual void scale(double* a, int n, double scale) = 0;

    virtual void daxpby(int n, double* a, double alpha, double* b, double beta) = 0;

    virtual ~IBlas() = default;
};

#endif //MLP_IBLAS_H
