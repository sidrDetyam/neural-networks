
#include "CpuBlas.h"
#include "Tensor.h"
#include <vector>
#include <iostream>
#include "Conv2dNaive.h"
#include "CsvDataLoader.h"
#include "Conv2d.h"

using namespace std;
using namespace nn;

int main() {
    Tensor tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}, {5, 4});
    Tensor out({54});
    cout << out.data().size() << endl;
    Conv2d::img2col(tensor.data().data(), 5, 4, 3, out.data().data());

    cout << tensor << "\n\n";
    cout << out << endl;
}