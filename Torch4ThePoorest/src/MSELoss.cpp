//
// Created by sidr on 21.05.23.
//

#include "MSELoss.h"
#include "Utils.h"

std::pair<double, nn::Tensor> nn::MSELoss::apply(const nn::Tensor &model_output,
                                                 const nn::Tensor &correct_output){
    const auto& shape = model_output.get_shape();
    ASSERT_RE(model_output.isSameShape(correct_output));
    ASSERT_RE(shape.size()==2 && shape[0]>0 && shape[1]>0);

    double loss = 0;
    Tensor grad = Tensor(model_output.get_shape());

    for(size_t b=0; b<shape[0]; ++b){
        double loss_local = 0;
        for(size_t i=0; i<shape[1]; ++i){
            const double diff = model_output({b, i}) - correct_output({b, i});
            loss_local += -2./diff * diff;

            grad({b, i}) = -2./(double)shape[1] * diff;
        }
        loss += loss_local;
    }

    return {loss/double(shape[0]), std::move(grad)};
}
