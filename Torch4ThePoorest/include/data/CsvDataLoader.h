//
// Created by sidr on 10.04.23.
//

#ifndef TORCH4THEPOOREST_CSVDATALOADER_H
#define TORCH4THEPOOREST_CSVDATALOADER_H

#include <string>
#include <vector>
#include "csv.hpp"
#include "IDataLoader.h"

namespace nn{

    class CsvDataLoader: public IDataLoader{
    public:
        explicit CsvDataLoader(int batch_size,
                               bool last_incomplete,
                               const std::string& fname,
                               size_t feature_size,
                               std::vector<size_t> targets);

        batch_t next_batch() override;

        bool has_next() override;

        std::vector<batch_t> read_all() override;

    private:
        csv::CSVReader reader_;
        std::vector<size_t> target_cols_;
        const int batch_size_;
        const bool last_incomplete_;
        const size_t feature_size_;
    };
}

#endif //TORCH4THEPOOREST_CSVDATALOADER_H
