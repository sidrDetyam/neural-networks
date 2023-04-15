//
// Created by sidr on 15.04.23.
//

#ifndef TORCH4THEPOOREST_ISERIALIZABLE_H
#define TORCH4THEPOOREST_ISERIALIZABLE_H

class ISerializable{
public:
//    template<class InputIterator>
//    virtual void serialize(InputIterator& iterator) = 0;

    virtual void deserialize() = 0;

    virtual ~ISerializable() = default;
};

#endif //TORCH4THEPOOREST_ISERIALIZABLE_H
