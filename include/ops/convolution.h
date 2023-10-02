#include <assert.h>

#include "tensor.h"
#include "shape.h"

#pragma once

using namespace nninfer::tensor;

namespace nninfer{
    
namespace ops{


template<typename T>
inline T get_padded_data(const T* data, 
           const int x,
           const int y,
           const int w,
           const int h,
           const int padding_x,
           const int padding_y) {

    int x_ = x - padding_x;
    int y_ = y - padding_y;
    if(x_ < 0 || y_ < 0 || x_ >= w || y_ >= h) {
        return static_cast<T>(0);
    }

    return data[x_ + y_ * w];
}

template<typename T>
inline void conv2d_naive_single(const Tensor<T> &input, 
                         const Tensor<T> &weight,
                         const Tensor<T> &output,
                         const int stride_x,
                         const int stride_y,
                         const int padding_x,
                         const int padding_y,
                         const int use_bias) {
    // assert the dimension of input must be 2
    assert(input.shape().dim == 2 && "The dimension of input must be 2 for single channel 2d-convolution");

    T* input_data = input.data();
    T* kernel_data = weight.data();
    T* output_data = output.data();

    int iw = input.shape().shape[0];
    int ih = input.shape().shape[1];
    int kw = weight.shape().shape[0];
    int kh = weight.shape().shape[1];
    int ow = output.shape().shape[0];
    int oh = output.shape().shape[1];

    for(int i = 0; i < oh; i++) {
        for(int j = 0; j < ow; j++) {
            int y = i * stride_y;
            int x = j * stride_x;
            T sum = 0;
            for(int m = 0; m < kh; m++) {
                for(int n = 0; n < kw; n++) {
                    sum += get_padded_data(input_data, x + n, y + m, iw, ih, padding_x, padding_y) * kernel_data[n + m * kw];
                }
            }
            output_data[j + i * ow] += sum;
        }
    }
}    

template<typename T>
void conv2d_naive(const Tensor<T> &input, 
            const Tensor<T> &weight, 
            Tensor<T> &output, 
            const int stride_x,
            const int stride_y, 
            const int padding_x,
            const int padding_y, 
            const int groups,
            const int use_bias) {

    // assert the dimension of input must be 3
    assert(input.shape().dim == 3 && "The dimension of input must be 3 for 2d-convolution");
    
    // T* input_data = input.data();
    // T* kernel_data = weight.data();
    // T* output_data = output.data();

    Shape input_shape = input.shape();
    Shape weight_shape = weight.shape();

    int num_channel = input_shape.shape[0];
    int ih = input_shape.shape[1];
    int iw = input_shape.shape[2];

    int num_kernel = weight_shape.shape[0];
    int kh = weight_shape.shape[1];
    int kw = weight_shape.shape[2];

    if (groups > 1) {

    }else {
        for(int i = 0; i < num_kernel; i++) {
            Tensor<T> subWeight = weight.sub(i);
        }
    }


}
    
} // namespace ops

} // namespace nninfer