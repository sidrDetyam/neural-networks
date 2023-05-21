//
// Created by sidr on 22.05.23.
//

#include "lab3.h"
#include "MAELoss.h"

using namespace std;
using namespace nn;

int main() {

    string test_path = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/data/test_lab3.csv";
    string model_dump_path = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/Torch4ThePoorest/data/lab3_model";

    nn::CsvDataLoader loader_test(64, true,
                                  test_path,
                                  seq_len*input_size+1, {0});

    nn::Sequential rnn = rnn_model();

    {
        ifstream fin(model_dump_path);
        fin >> rnn;
    }

    CachingDataLoader test(loader_test);

    cout << "Loaded\n\n";

    //nn::MSELoss loss;
    MAELoss loss;
    

    Tensor out_;
    Tensor correct_;

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

        total += 1;//(int) out.getBsize();
        err += l.first;

        bi = tqdm.next();

//            if(tqdm.is_end()){
////                cout<< out.get_shape().size() << std::endl;
//            }
    }

    cout << "\nMean MAE per sample: " << err / total << endl << endl;

    for(size_t i=0; i<out_.get_shape()[0]; ++i){
        std::cout << i+1 << ") " << out_({i, 0}) << " " << correct_({i, 0}) << std::endl;
    }

    return 0;
}