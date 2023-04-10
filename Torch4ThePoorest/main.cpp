
#include "CpuBlas.h"
#include "Tensor.h"
#include <vector>
#include <iostream>
#include "Conv2d.h"

using namespace std;

int main() {

    auto blas = CpuBlas();
    vector<double> a{1, 1, 1, 2, 3,
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
                     4, 4, 0, 4, 4};

    vector<double> w{1, 0, -1,
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
                     1, 2, 3};

    Conv2d conv2D(2, 2, 3, 3, make_unique<CpuBlas>(), w);

    auto a_ = a;
    for(auto i : a){
        a_.push_back(i);
    }

    Tensor input(a, {1, 2, 6, 5});

    Tensor output = conv2D.forward(std::move(input));

    Tensor loss(std::vector<double>(24, 1.), {1, 2, 4, 3});
    Tensor g = conv2D.backward(loss);

    cout << g << endl;
}