//
// Created by sidr on 19.03.23.
//

#ifndef MLP_UTILS_H
#define MLP_UTILS_H

#include <iostream>

#define ASSERT(cond__) \
do{\
    if(!(cond__)){     \
        std::cerr << #cond__ << std::endl;     \
        std::abort(); \
    } \
}while(0)

#endif //MLP_UTILS_H
