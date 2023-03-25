#include <iostream>
#include <cblas.h>
#include "include/Batch.h"
#include <cstring>
#include <random>
#include "Model.h"
#include "Utils.h"
#include "ReLU.h"
#include "CrossEntropyLoss.h"

#include <unistd.h>

#include "include/LinearLayer.h"
#include "include/CpuBlas.h"

#include "csv.hpp"

using namespace std;
using namespace csv;


using data_t = std::vector<std::pair<std::vector<double>, int>>;

std::vector<std::pair<std::vector<double>, int>> load_data(){

    std::vector<std::pair<std::vector<double>, int>> data;

    CSVReader reader("/home/sidr/PycharmProjects/neural-networks/MLP/data/mush2.csv");
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


using batch_label_t = std::pair<Batch, std::vector<int>>;

std::vector<batch_label_t> get_batches(const data_t &data, int bsize){
    Batch batch(bsize, 45);
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


Model getMushModel(){

    std::vector<std::unique_ptr<ILayer>> layers;
    layers.emplace_back(linearLayerCreator(45, 45));
    layers.emplace_back(new ReLU());
    layers.emplace_back(linearLayerCreator(45, 16));
    layers.emplace_back(new ReLU());
    layers.emplace_back(linearLayerCreator(16, 8));
    layers.emplace_back(new ReLU());
    layers.emplace_back(linearLayerCreator(8, 2));

    Model model(std::move(layers));

    return model;
}


enum CONSTS{
    BATCH_SIZE = 64,
    EPOCHS = 10
};


int main() {

    auto data = load_data();
    auto train_test = split(data, 0.8);
    auto train_batches = get_batches(train_test.first, BATCH_SIZE);
    auto test_batches = get_batches(train_test.first, BATCH_SIZE);

    Model model = getMushModel();
    CrossEntropyLoss loss;

    for(int e=0; e<EPOCHS; ++e){
        double err = 0;

        for(const auto & batch : test_batches){
            Batch b = batch.first;
            auto out = model.forward(std::move(b));
            auto l = loss.apply(out, batch.second);
            model.backward(l.second);
            model.step();

            err += l.first;
        }

        cout << e << " " << err / (double)BATCH_SIZE << endl;
    }

    return 0;
}
