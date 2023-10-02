#include <exception>

#include "tensor.h"
#include "base_layer.h"

using namespace nninfer::tensor;

namespace nninfer
{

namespace layer
{
template <typename T>
void BaseLayer<T>::forward(const Tensor<T> &input, Tensor<T> &output)
{
    throw std::exception("Not implemented");
}
    
} // namespace layer

    
} // namespace nninfer
