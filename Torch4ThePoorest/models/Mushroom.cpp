//
// Created by sidr on 26.03.23.
//

#include <iostream>
#include "Tensor.h"
#include <cstring>
#include <random>
#include "Sequential.h"
#include "Utils.h"
#include "ReLU.h"
#include "CrossEntropyLoss.h"
#include "SgdOptimizerCreator.h"

#include <unistd.h>

#include "Linear.h"
#include "CpuBlas.h"

#include "csv.hpp"

using namespace std;
using namespace csv;
using namespace nn;


using data_t = std::vector<std::pair<std::vector<double>, int>>;

static std::vector<std::pair<std::vector<double>, int>> load_data(){

    std::vector<std::pair<std::vector<double>, int>> data;

    CSVReader reader("/home/sidr/PycharmProjects/neural-networks/Torch4ThePoorest/data/mush2.csv");
    for(auto& row : reader){
        std::vector<double> v;
        for(auto & c : row){
            v.push_back(c.get<double>());
        }
        v.pop_back();
        int cl = std::abs(v.back()) > 0.001;
        data.emplace_back(v, cl);
    }

    return data;
}

std::pair<data_t, data_t> split(data_t data, double train_fraction = 0.8){
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);

    const int train_size = static_cast<int>((double)data.size() * train_fraction);
    data_t train_data(data.begin(), data.begin() + train_size);
    data_t test_data(data.begin() + train_size, data.end());

    return {train_data, test_data};
}


using batch_label_t = std::pair<Tensor, std::vector<int>>;

std::vector<batch_label_t> get_batches(const data_t &data, int bsize){
    Tensor batch({static_cast<size_t>(bsize), 45});
    std::vector<int> classes(bsize);
    int pos = 0;

    std::vector<batch_label_t> batches;

    while(pos + bsize <= data.size()){
        for(int i=0; i<bsize; ++i){
            std::memcpy(batch[i], data[pos+i].first.data(), 45 * sizeof(double));
            classes[i] = data[pos+i].second;
        }

        batches.emplace_back(batch, classes);
        pos += bsize;
    }

    return batches;
}


LinearLayer* linearLayerCreator(size_t input, size_t output){
    return new LinearLayer(input, output,
                           random_vector_gauss(input * output, 0, 1),
                           random_vector_gauss(output, 0, 1),
                           std::make_unique<CpuBlas>());
}


Model getMushModel2(){
    std::vector<std::unique_ptr<ILayer>> layers;
    layers.emplace_back(linearLayerCreator(45, 22));
    layers.emplace_back(new ReLU(CpuBlas::of()));
    layers.emplace_back(linearLayerCreator(22, 11));
    layers.emplace_back(new ReLU(CpuBlas::of()));
    layers.emplace_back(linearLayerCreator(11, 2));

    SgdOptimizerCreator sgd_creator(0.95, 0.05, std::make_shared<CpuBlas>());

    Model model(std::move(layers), std::make_unique<SgdOptimizerCreator>(std::move(sgd_creator)));

    return model;
}


enum CONSTS{
    BATCH_SIZE = 64,
    EPOCHS = 10
};


std::pair<double, double> loss_accuracy(Model& model, IClassificationLostFunction& loss, const std::vector<batch_label_t>& test){
    double err = 0;
    int correct = 0;

    for(const auto & batch : test){
        Tensor b = batch.first;
        auto out = model.forward(std::move(b));
        auto l = loss.apply(out, batch.second);

        for(size_t i = 0; i < out.getBsize(); ++i){
            long cl = std::max_element(out[i], out[i+1]) - out[i];
            correct += cl == batch.second[i];
        }

        err += l.first;
    }

    return {err / (double) BATCH_SIZE, correct / (double)(BATCH_SIZE * test.size())};
}


int main() {
    auto data = load_data();
    auto train_test = split(data, 0.7);
    auto train_batches = get_batches(train_test.first, BATCH_SIZE);
    auto test_batches = get_batches(train_test.first, BATCH_SIZE);

    auto all_batches = get_batches(data, BATCH_SIZE);

    Model model = getMushModel2();
    CrossEntropyLoss loss;

    for(int e=0; e<EPOCHS; ++e){
        double err = 0;

        for(const auto & batch : train_batches){
            Tensor b = batch.first;
            auto out = model.forward(std::move(b));
            auto l = loss.apply(out, batch.second);
            model.backward(l.second);
            model.step();

            err += l.first;

//            auto bruh = loss_accuracy(model, loss, test_batches);
//            cout << bruh.first << " " << bruh.second << " " << endl;
//            sleep(1);
        }

        auto val = loss_accuracy(model, loss, test_batches);
        cout << e << " " << err / (double)BATCH_SIZE << " " << val.first << " " << val.second <<
             " " << loss_accuracy(model, loss, all_batches).second << endl;
        sleep(1);
    }

    return 0;
}
