
#include "CpuBlas.h"
#include "Tensor.h"
#include <vector>
#include <iostream>

using namespace std;

int main() {

    auto blas = CpuBlas();
    vector<double> a{1, 1, 1, 2, 3,
                     1, 1, 1, 2, 3,
                     1, 1, 1, 2, 3,
                     2, 2, 2, 2, 3,
                     3, 3, 3, 3, 3,
                     4, 4, 4, 4, 4};

    vector<double> w{1, 0, -1,
                     2, 0, -2,
                     1, 0, -1};

    vector<double> w_t{1, 2, 1,
                       0, 0, 0,
                       -1, -2, -1};

    vector<double> c(4 * 3);
    blas.convolve(a.data(), w.data(), c.data(), 6, 5, 3, 3, 0);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            cout << c[3 * i + j] << " ";
        }
        cout << endl;
    }
}