
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
#include <ISerializable.h>

using namespace std;
using namespace nn;

template<class Iterator>
void foo(Iterator& it){
    for(int i=0; i<5; ++i){
        double val = *it;
        cout << val << endl;
        ++it;
    }
}



#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

class MyClass {
public:
    int x;
    double y;
    std::string name;

    vector<double> bruh = {1, 2, 3, 4, 5};

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & x;
        ar & y;
        ar & name;
        ar & BOOST_SERIALIZATION_NVP(bruh);
    }

    //BOOST_SERIALIZATION_SPLIT_MEMBER();
};

int main() {
//    // создаем объект класса
    MyClass obj;
    obj.x = 10;
    obj.y = 3.14;
    obj.name = "My Object";
//
//    // сериализуем объект в файл
    string s = "/media/sidr/6C3ED7833ED7452C/bruh/PycharmProjects/neural-networks/Torch4ThePoorest/data/ser";

    std::ofstream ofs(s);
    boost::archive::text_oarchive oa(ofs);
    oa << obj;

    obj.name = "Bruh_obk";
//    oa << obj;

//    return 0;

    // десериализуем объект из файла
    MyClass obj2;
    std::ifstream ifs(s);
    boost::archive::text_iarchive ia(ifs);
    ia >> obj2;
    ia >> obj2;

    // проверяем, что объекты идентичны
    assert(obj.x == obj2.x);
    assert(obj.y == obj2.y);
    assert(obj.name == obj2.name);

    return 0;
}
