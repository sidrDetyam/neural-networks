//
// Created by sidr on 15.04.23.
//

#ifndef TORCH4THEPOOREST_TQDM_H
#define TORCH4THEPOOREST_TQDM_H

#include <sstream>
#include <iostream>

class Tqdm{
public:

    explicit Tqdm(int wide);

    int start(int last);

    int next();

    [[nodiscard]] bool is_end() const;

    void print_promnt();

    template<class T>
    friend Tqdm &operator<<(Tqdm &os, T t);

private:
    int wide_;
    int last_ = 0;
    int curr_ = 0;
    int len_ = 0;
};

template<class T>
Tqdm &operator<<(Tqdm &os, T t){
    std::stringstream ss;
    ss.setf(std::ios::fixed);
    ss.precision(2);
    ss << t;
    os.len_ += (int)ss.str().size();
    std::cout << ss.str();
    std::flush(std::cout);
    return os;
}

#endif //TORCH4THEPOOREST_TQDM_H
