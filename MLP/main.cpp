#include <iostream>
#include <cblas.h>
#include "include/Batch.h"
#include <cstring>

#include "include/LinearLayer.h"
#include "include/CpuBlas.h"

using namespace std;

int main() {

    LinearLayer ll(3, 2, CpuBlas::of());

    Batch b({{1, 2, 3}});

    Batch bb({{90, 168}});
    auto out = ll.forward(b);
    auto ing = ll.backward(bb);


    return 0;
}
