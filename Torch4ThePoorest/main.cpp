
#include "CpuBlas.h"
#include "Tensor.h"
#include <vector>
#include <iostream>
#include "Conv2dNaive.h"
#include "CsvDataLoader.h"
#include "Conv2d.h"
#include "Utils.h"

using namespace std;
using namespace nn;

int main() {
    const vector<double> a{1, 1, 1, 2, 3,
                           1, 1, 1, 2, 3,
                           1, 1, 1, 2, 3,
                           2, 2, 2, 2, 3,
                           3, 3, 3, 3, 3,
                           4, 4, 4, 4, 4,

                           1, 1, 5, 2, 3,
                           1, 1, 6, 2, 3,
                           1, 1, 7, 2, 3,
                           2, 2, 8, 2, 3,
                           3, 3, 9, 3, 3,
                           4, 4, 0, 4, 4

    };

    const vector<double> w{1, 0, -1,
                           2, 0, -2,
                           1, 0, -1,
                           1, 3, -1,
                           2, 44, -2,
                           1, 2, -1,

                           3, 4, 5,
                           6, 7, 8,
                           9, 10, 11,
                           11, 23, 42,
                           2, 2, 2,
                           1, 2, 3
    };

    Conv2dNaive conv2DNaive(2, 2, 3, 3, make_unique<CpuBlas>(), w);
    Conv2d conv2D(2, 2, 3, make_unique<CpuBlas>(), w);

    Tensor input(a, {1, 2, 6, 5});

    Tensor loss(vector<double>(2 * 4 * 3, 1), {1, 2, 4, 3});

    conv2DNaive.forward(Tensor(input));
    conv2D.forward(Tensor(input));

    conv2D.backward(Tensor(loss));
    conv2DNaive.backward(Tensor(loss));

    Tensor c1(conv2DNaive.getParametersGradient(), {2, 2, 3, 3});
    Tensor c2(conv2D.getParametersGradient(), {2, 2, 3, 3});

    cout << c1 << endl;

    cout << endl << c2 << endl;

    cout << is_same_vectors(c1.data(), c2.data(), 1e-5);

    return 0;
}