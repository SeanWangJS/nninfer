#include <variant>

#include "layer/pool_layer.h"
#include "ops/max_pool.h"

using namespace nninfer::ops;

namespace nninfer
{

namespace layer
{

template class MaxPool2d<float>;

template <typename T>
MaxPool2d<T>::MaxPool2d(std::variant<std::pair<int, int>, int> kernel_size,
                        std::variant<std::pair<int, int>, int> stride,
                        std::variant<std::pair<int, int>, int> padding,
                        std::variant<std::pair<int, int>, int> dilation,
                        bool ceil_mode)
{
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
    if(dilation.index() == 0) {
        this->dilation = std::get<std::pair<int, int>>(dilation);
    }else {
        int d = std::get<int>(dilation);
        this->dilation = std::make_pair(d, d);
    }
    this->ceil_mode = ceil_mode;
}
    
template<typename T>
void MaxPool2d<T>::forward(const Tensor<T> &input, Tensor<T> &output)
{
    max_pool2d(input, output, 
               this->kernel_size.first, this->kernel_size.second,
               this->stride.first, this->stride.second,
               this->padding.first, this->padding.second);
}

template<typename T>
Tensor<T> MaxPool2d<T>::forward(const Tensor<T> &input)
{
    int iw = input.shape()[2];
    int ih = input.shape()[3];
    int ow = (iw + 2 * padding.first - kernel_size.first) / stride.first + 1;
    int oh = (ih + 2 * padding.second - kernel_size.second) / stride.second + 1;

    Shape output_shape = Shape({input.shape()[0], input.shape()[1], ow, oh});
    Tensor<T> output = Tensor<T>::zeros(output_shape);

    max_pool2d(input, output, 
               this->kernel_size.first, this->kernel_size.second,
               this->stride.first, this->stride.second,
               this->padding.first, this->padding.second);

    return output;
}

} // namespace layer

    
} // namespace nninfer
