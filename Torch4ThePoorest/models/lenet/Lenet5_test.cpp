//
// Created by sidr on 16.04.23.
//

#include "Lenet5.h"
#include "CsvDataLoader.h"
#include "CrossEntropyLoss.h"
#include "CachingDataLoader.h"
#include "TransformDataLoaderDecorator.h"
#include "Classification.h"

using namespace nn;
using namespace std;

int main(){
    const string train_path = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/Torch4ThePoorest/data/mnist_train.csv";
    const string test_path = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/Torch4ThePoorest/data/mnist_test.csv";
    const string model_dump_path = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/Torch4ThePoorest/data/lenet5_model";

    //auto model = l_model();
    auto model = lenet5_model();
    {
        ifstream fin(model_dump_path);
        fin >> model;
    }

    nn::CsvDataLoader loader_test(130, true,
                                  test_path,
                                  785, {0});

    CrossEntropyLoss loss;
    auto normalization = [](batch_t &b){
        for (auto &i: b.first.data()) {
            i /= 255.;
        }
    };

    CachingDataLoader test(TransformDataLoaderDecorator(loader_test, normalization));
    auto test_result = classification_test(model, 10, test, loss, true);
    cout << "  Testing result: Mean loss: " << test_result.first << ", Accuracy: " << accuracy(test_result.second) << endl;

    return 0;
}