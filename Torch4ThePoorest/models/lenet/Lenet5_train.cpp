//
// Created by sidr on 10.04.23.
//

#include "CpuBlas.h"
#include <vector>
#include <iostream>
#include "Conv2d.h"
#include "CsvDataLoader.h"
#include "Sequential.h"
#include "Linear.h"
#include "AvgPolling.h"
#include "Reshaper.h"
#include "SgdOptimizerCreator.h"
#include "CrossEntropyLoss.h"
#include "Dropout.h"
#include "Tanh.h"
#include "Tqdm.h"
#include "WeightInitializers.h"
#include "TransformDataLoaderDecorator.h"
#include "CachingDataLoader.h"
#include "Classification.h"
#include "Lenet5.h"

using namespace nn;
using namespace std;

int main() {

    string train_path = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/Torch4ThePoorest/data/mnist_train.csv";
    string test_path = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/Torch4ThePoorest/data/mnist_test.csv";
    string model_dump_path = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/Torch4ThePoorest/data/lenet5_model";
    const int epochs = 1;

    nn::CsvDataLoader loader_train(130, true,
                                   train_path,
                                   785, {0});

    nn::CsvDataLoader loader_test(130, true,
                                  test_path,
                                  785, {0});

    nn::Sequential lenet = lenet5_model();
    //nn::Sequential lenet = l_model();


    auto normalization = [](batch_t &b){
        for (auto &i: b.first.data()) {
            i /= 255.;
        }
    };

    CachingDataLoader train(TransformDataLoaderDecorator(loader_train, normalization));
    CachingDataLoader test(TransformDataLoaderDecorator(loader_test, normalization));

    cout << "Loaded\n\n";

    nn::CrossEntropyLoss loss;

    for (int e = 0; e < epochs; ++e) {
        train.shuffle();
        double err = 0;
        int total = 0;
        cout << "Epoch " << e+1 << endl;

        Tqdm tqdm(3);
        for (int bi = tqdm.start((int)train.size()); !tqdm.is_end();) {
            const auto batch = train.next_batch();
            Tensor input = batch.first;

            auto out = lenet.forward(std::move(input));
            auto l = loss.apply(out, to_one_hot(batch.second));
            lenet.backward(l.second);
            lenet.step();

            total += (int) out.getBsize();
            err += l.first;

            bi = tqdm.next();
            CachingDataLoader bruh({batch});
            tqdm << "  Training... " << bi << "/" << train.size()
                 << " Mean loss(epoch): " << err / total
                 << ", Mean loss(batch): " << l.first / (double) l.second.getBsize()
                 << ", Accuracy(batch): " << accuracy(classification_test(lenet, 10, bruh, loss).second);
        }

        test.reset();
        cout << endl;
        auto test_result = classification_test(lenet, 10, test, loss, true);
        cout << "  Testing result: " << err / total << " " << test_result.first << " " << accuracy(test_result.second) << endl;
        sleep(5);
    }

    ofstream fout(model_dump_path);
    fout << lenet;
}
