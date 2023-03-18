//
// Created by sidr on 18.03.23.
//

#ifndef MLP_IBLAS_H
#define MLP_IBLAS_H

class IBlas{
public:
    virtual void dgemm(const double* a, const double *b, bool isATransposed, bool isBTransposed,
                       double *c, int m, int n, int k, double beta) = 0;

    virtual ~IBlas() = default;
};

#endif //MLP_IBLAS_H
