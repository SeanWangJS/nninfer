#include <variant>

#include "layer/pool_layer.h"


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
    throw std::exception("Not implemented");
}

template<typename T>
Tensor<T> MaxPool2d<T>::forward(const Tensor<T> &input)
{
    return input;
}

} // namespace layer

    
} // namespace nninfer
