#include <iostream>
#include <cblas.h>
#include "include/Batch.h"
#include <cstring>

#include "include/LinearLayer.h"
#include "include/CpuBlas.h"

using namespace std;

int main() {

    LinearLayer ll(3, 2, CpuBlas::of());

    return 0;
}
