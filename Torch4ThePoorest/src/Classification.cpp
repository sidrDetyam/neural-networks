//
// Created by sidr on 15.04.23.
//

#include "Classification.h"
#include "Tqdm.h"
#include "Utils.h"

namespace {
    std::vector<int> to_one_hot(const nn::Tensor &tensor){
        ASSERT_RE(tensor.get_shape().size() == 2 && tensor.get_shape()[1] == 1);
        std::vector<int> one_hot;
        for(auto i : tensor.data()){
            one_hot.push_back((int) i);
        }
        return one_hot;
    }
}

std::pair<double, nn::Tensor> nn::classification_test(Sequential &model,
                                                      int cnt_of_classes,
                                                      IDataLoader &data_loader,
                                                      const IClassificationLostFunction &loss) {

    Tensor table({(size_t) cnt_of_classes, 4});
    auto test_batches = data_loader.read_all();
    Tqdm tqdm(3);

    double err = 0.;
    size_t cnt = 0;

    for (int i = tqdm.start((int) test_batches.size()); !tqdm.is_end();) {

        tqdm << "  Testing " << i << "/" << test_batches.size();

        nn::Tensor batch = std::move(test_batches[i].first);
        const auto output = model.forward(std::move(batch));
        const std::vector<int> one_hot = to_one_hot(test_batches[i].second);
        const auto l = loss.apply(output, one_hot);

        for (size_t j = 0; j < output.getBsize(); ++j) {
            const long predicted = std::max_element(output[j], output[j + 1]) - output[j];
            const long correct = one_hot[j];
            if(predicted == correct){
                table[predicted][0] += 1;
                for(size_t k = 0; k < cnt_of_classes; ++k){
                    if(k != predicted){
                        table[k][2] += 1;
                    }
                }
            }
            else{
                table[predicted][1] += 1;
                table[correct][3] += 1;
                for(size_t k = 0; k < cnt_of_classes; ++k){
                    if(k != predicted && k != correct){
                        table[k][2] += 1;
                    }
                }
            }
        }

        err += l.first;
        cnt += l.second.getBsize();
        i = tqdm.next();
    }

    return {err / (double)cnt, table};
}

double nn::accuracy(const nn::Tensor &table) {
    ASSERT_RE(table.get_shape().size() == 2 && table.get_shape()[0] > 0 && table.get_shape()[1] == 4);

    long all = 0;
    long tp = 0;
    for(int i = 0; i<table.getBsize(); ++i){
        all += (long) (table[i][0] + table[i][1]);
        tp += (long) table[i][0];
    }

    return (double) tp / (double) all;
}
