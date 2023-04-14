//
// Created by sidr on 15.04.23.
//

#include <cmath>
#include "Tqdm.h"

using namespace std;

Tqdm::Tqdm(int wide): wide_(wide){}

int Tqdm::start(int last){
    last_ = last;
    len_ = 0;
    curr_ = 0;
    print_promnt();
    return curr_;
}

int Tqdm::next(){
    ++curr_;
    print_promnt();
    return curr_;
}

[[nodiscard]] bool Tqdm::is_end() const{
    return curr_ == last_;
}

void Tqdm::print_promnt(){
    stringstream ss("");

    for(int i=0; i<len_; ++i){
        cout << "\010 \010";
    }

    ss << '[';

    const int p = std::ceil(wide_ * 10. * curr_ / last_);
    for(int i=0; i<p; ++i){
        ss << '#';
    }
    for(int i=0; i<wide_ * 10 - p; ++i){
        ss << ' ';
    }

    ss << "] ";
    ss << 100. * curr_ / last_ << "%";

    len_ = (int)ss.str().size();
    cout << "\x1B[32m";
    cout << ss.str();
    cout << "\x1B[0m";
    std::flush(cout);
}