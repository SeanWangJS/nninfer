#include "tensor.h"
#include "layer/bn_layer.h"
#include "ops/batch_norm.h"

using namespace nninfer::tensor;

namespace nninfer
{

namespace layer
{

template class BatchNorm2d<float>;

template <typename T>
BatchNorm2d<T>::BatchNorm2d(Tensor<T> running_mean,
                            Tensor<T> running_var,
                            Tensor<T> weight,
                            Tensor<T> bias,
                            float eps)
{
    this->running_mean = running_mean;
    this->running_var = running_var;
    this->weight = weight;
    this->bias = bias;
    this->eps = eps;
    
}

template <typename T>
void BatchNorm2d<T>::forward(const Tensor<T> &input, 
                                 Tensor<T> &output)
{
    ops::batch_norm(input, running_mean, running_var, weight, bias, eps);
    output = input;
}

template <typename T>
Tensor<T> BatchNorm2d<T>::forward(const Tensor<T> &input)
{
    ops::batch_norm(input, running_mean, running_var, weight, bias, eps);
    return input;
}

} // namespace layer

    
} // namespace nninfer
