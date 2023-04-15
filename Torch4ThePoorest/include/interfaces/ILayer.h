//
// Created by sidr on 18.03.23.
//

#ifndef MLP_ILAYER_H
#define MLP_ILAYER_H

#include "Tensor.h"
#include <vector>
#include <boost/serialization/nvp.hpp>

namespace nn {

    class ILayer {
    public:
        virtual Tensor forward(Tensor &&input) = 0;

        virtual Tensor backward(const Tensor &output) = 0;

        virtual std::vector<double> &getParametersGradient();

        virtual std::vector<double> &getParameters();

        virtual ~ILayer() = default;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & BOOST_SERIALIZATION_NVP(params_);
        }

    protected:
        std::vector<double> params_;
        std::vector<double> grad_;
    };
}

#endif //MLP_ILAYER_H
