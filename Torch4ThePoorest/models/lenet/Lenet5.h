//
// Created by sidr on 16.04.23.
//

#ifndef TORCH4THEPOOREST_LENET5_H
#define TORCH4THEPOOREST_LENET5_H

#include "Linear.h"
#include "Conv2d.h"
#include "Sequential.h"

namespace nn {

    Linear *linearLayerCreator(size_t input, size_t output);

    Conv2d *conv2DCreator(size_t in_channels, size_t out_channels);

    Sequential lenet5_model();

    Sequential l_model();

} // nn

#endif //TORCH4THEPOOREST_LENET5_H
