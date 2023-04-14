
#include "CpuBlas.h"
#include "Tensor.h"
#include <vector>
#include <iostream>
#include "Conv2dNaive.h"
#include "CsvDataLoader.h"
#include "Conv2d.h"
#include "Utils.h"
#include <sstream>
#include <unistd.h>

using namespace std;
using namespace nn;

class Tqdm{
public:

    explicit Tqdm(int wide): wide_(wide){}

    int start(int last){
        last_ = last;
        len_ = 0;
        curr_ = 0;
        print_promnt();
        return curr_;
    }

    int next(){
        ++curr_;
        print_promnt();
        if(is_end()){
            cout << "\n";
        }
        return curr_;
    }

    [[nodiscard]] bool is_end() const{
        return curr_ == last_;
    }

    void print_promnt(){
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
        cout << ss.str();
        std::flush(cout);
    }

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
    cout << t;
    ss << t;
    os.len_ += (int)ss.str().size();
    std::flush(cout);
    return os;
}


int main() {

    Tqdm tqdm (5);
    for(int i = tqdm.start(100); !tqdm.is_end(); i = tqdm.next()){
        tqdm << " bruh" << " " << i;
        sleep(1);
    }

    return 0;
}