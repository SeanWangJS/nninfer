#include <variant>

#include "tensor.h"
#include "layer/conv_layer.h"
#include "ops/convolution.h"

using namespace nninfer::ops;

namespace nninfer
{
    
namespace layer
{

template class Conv2d<float>;

template <typename T>
Conv2d<T>::Conv2d(int in_channels,
               int out_channels,
               std::variant<std::pair<int,int>, int> kernel_size,
               std::variant<std::pair<int,int>, int> stride,
               std::variant<std::pair<int,int>, int> padding,
               int groups,
               bool use_bias) {
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    if(kernel_size.index() == 0) {
        this->kernel_size = std::get<std::pair<int, int>>(kernel_size);
    }else {
        int k = std::get<int>(kernel_size);
        this->kernel_size = std::make_pair(k, k);
    }

    if(stride.index() == 0) {
        this->stride = std::get<std::pair<int, int>>(stride);
    }else {
        int s = std::get<int>(stride);
        this->stride = std::make_pair(s, s);
    }

    if(padding.index() == 0) {
        this->padding = std::get<std::pair<int, int>>(padding);
    }else {
        int p = std::get<int>(padding);
        this->padding = std::make_pair(p, p);
    }

    this->groups = groups;

    // random initialize weight
    Shape weight_shape = Shape({out_channels, in_channels / groups, this->kernel_size.first, this->kernel_size.second});
    this->weight = Tensor<T>::random(weight_shape, -1, 1);

    // random initialize bias
    this->use_bias = use_bias;
    if(use_bias) {
        Shape bias_shape = Shape({out_channels});
        this->bias = Tensor<T>::random(bias_shape, -1, 1);
    }
}

template <typename T>
void Conv2d<T>::forward(const Tensor<T> &input, Tensor<T> &output) {

    int stride_x = this->stride.first;
    int stride_y = this->stride.second;
    int padding_x = this->padding.first;
    int padding_y = this->padding.second;
    int groups = this->groups;
    Tensor<T> weight = this->weight;
    conv2d_batch(input, weight, output, stride_x, stride_y, padding_x, padding_y, groups);

}

// template <typename T>
// Tensor<T> Conv2d<T>::forward(const Tensor<T> &input) {

//     int stride_x = this->stride.first;
//     int stride_y = this->stride.second;
//     int padding_x = this->padding.first;
//     int padding_y = this->padding.second;
//     int groups = this->groups;
//     Tensor<T> weight = this->weight;

//     Tensor<T> output = Tensor<T>::zeros({this->out_channels, 1, 1});
//     conv2d_naive(input, weight, output, stride_x, stride_y, padding_x, padding_y, groups);
//     return output;


// }

} // namespace layer

} // namespace nninfer
