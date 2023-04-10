
#include "CpuBlas.h"
#include "Tensor.h"
#include <vector>
#include <iostream>
#include "Conv2d.h"
#include "CsvDataLoader.h"

using namespace std;
using namespace nn;

int main() {
    nn::CsvDataLoader loader(2, true,
                             "/home/sidr/PycharmProjects/neural-networks/Torch4ThePoorest/data/mnist_test.csv",
                             785, {0});

    auto b = loader.next_batch();
    cout << b.second << endl << endl;

    cout << b.first << endl;
}