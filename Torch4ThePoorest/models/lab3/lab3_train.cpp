//
// Created by sidr on 22.05.23.
//

#include "lab3.h"

using namespace std;
using namespace nn;

int main() {

    string train_path = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/data/train_lab3.csv";
    string test_path = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/data/test_lab3.csv";
    string model_dump_path = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/Torch4ThePoorest/data/lab3_model";
    const int epochs = 2000;

    nn::CsvDataLoader loader_train(64, true,
                                   train_path,
                                   seq_len*input_size+1, {0});

    nn::CsvDataLoader loader_test(64, true,
                                  test_path,
                                  seq_len*input_size+1, {0});

//    nn::CsvDataLoader loader_test(130, true,
//                                  test_path,
//                                  785, {0});

    //nn::Sequential rnn = lenet5_model();
    nn::Sequential rnn = rnn_model();

    CachingDataLoader train(loader_train);
    CachingDataLoader test(loader_test);
//    CachingDataLoader test(TransformDataLoaderDecorator(loader_test, normalization));

    cout << "Loaded\n\n";

    nn::MSELoss loss;

    Tensor out_;
    Tensor correct_;
    for (int e = 0; e < epochs; ++e) {
        train.shuffle();
        train.reset();
        double err = 0;
        int total = 0;
        cout << "Epoch " << e+1 << endl;

        Tqdm tqdm(3);
        for (int bi = tqdm.start((int)train.size()); !tqdm.is_end();) {
            const auto batch = train.next_batch();
            Tensor input = batch.first;

            Tensor out = rnn.forward(std::move(input));
            if(out.get_shape()[0] > 60){
                out_ = out;
                correct_ = batch.second;
            }
            auto l = loss.apply(out, batch.second);
            rnn.backward(l.second);
            rnn.step();

            total += 1;//(int) out.getBsize();
            err += l.first;

            bi = tqdm.next();

//            if(tqdm.is_end()){
////                cout<< out.get_shape().size() << std::endl;
//            }
        }

        cout << "Loss: " << err / total << endl;
        if(err / total < 200){
            break;
        }

//        sleep(5);
    }


    double err = 0;
    int total = 0;
    cout << "Testing " << endl;

    Tqdm tqdm(3);
    for (int bi = tqdm.start((int)test.size()); !tqdm.is_end();) {
        const auto batch = test.next_batch();
        Tensor input = batch.first;

        Tensor out = rnn.forward(std::move(input));
        if(out.get_shape()[0] > 60){
            out_ = out;
            correct_ = batch.second;
        }
        auto l = loss.apply(out, batch.second);
        rnn.backward(l.second);
        rnn.step();

        total += 1;//(int) out.getBsize();
        err += l.first;

        bi = tqdm.next();

//            if(tqdm.is_end()){
////                cout<< out.get_shape().size() << std::endl;
//            }
    }

    cout << "Loss: " << err / total << endl;

    std::cout << out_.size() << endl;
    std::cout << correct_.size() << endl;

    for(size_t i=0; i<out_.get_shape()[0]; ++i){
        std::cout << i+1 << ") " << out_({i, 0}) << " " << correct_({i, 0}) << std::endl;
    }

//    ofstream fout(model_dump_path);
//    fout << rnn;

    return 0;
}