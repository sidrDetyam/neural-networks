//
// Created by sidr on 18.03.23.
//

#ifndef MLP_IBLAS_H
#define MLP_IBLAS_H

class IBlas{
public:
    virtual void dgemm(const double* a, const double *b, double *c, int m, int n, int k) = 0;

    virtual ~IBlas() = default;
};

#endif //MLP_IBLAS_H
