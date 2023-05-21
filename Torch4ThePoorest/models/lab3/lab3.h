//
// Created by sidr on 21.05.23.
//

#ifndef TORCH4THEPOOREST_LAB3_H
#define TORCH4THEPOOREST_LAB3_H

#include <iostream>
#include "Linear.h"
#include "WeightInitializers.h"
#include "CpuBlas.h"
#include "Sequential.h"
#include "Reshaper.h"
#include "SgdOptimizerCreator.h"
#include "Rnn.h"
#include "Tanh.h"
#include "ReLU.h"
#include "CsvDataLoader.h"
#include "CachingDataLoader.h"
#include "MSELoss.h"
#include "Tqdm.h"

nn::Linear *linearLayerCreator(size_t input, size_t output);

const size_t seq_len = 20;
const size_t input_size = 17;

nn::Sequential rnn_model();

#endif //TORCH4THEPOOREST_LAB3_H
