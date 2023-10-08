#include "tensor.h"

#ifndef BASE_LAYER_H
#define BASE_LAYER_H

using namespace nninfer::tensor;

namespace nninfer
{

namespace layer
{

template<typename T>
class BaseLayer{

    virtual void forward(const Tensor<T> &input, 
                         Tensor<T> &output);

    virtual Tensor<T> forward(const Tensor<T> &input);
};
    
} // namespace layer

    
} // namespace nninfer

#endif // BASE_LAYER_H