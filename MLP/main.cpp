#include <iostream>
#include <cblas.h>
#include "include/Batch.h"
#include <cstring>

#include "include/LinearLayer.h"
#include "include/CpuBlas.h"

using namespace std;

int main() {

    std::vector<double> m({1., 2., 3., 4., 5, 6});

    LinearLayer ll(3, 2, std::move(m), {{11, 22}}, CpuBlas::of());

    Batch b({{1, 2, 3}});

    Batch bb({{90, 168}});
    auto out = ll.forward(std::move(b));
    auto ing = ll.backward(bb);
    auto pgrad = ll.getParametersGradient();

    return 0;
}
