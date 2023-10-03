#include <assert.h>
#include <iostream>

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
                         const int padding_y) {
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

/**
 * :param input: input tensor, shape is [C_in, ih, iw]
 * :param weight: weight tensor, shape is [C_out, C_in, kh, kw]
 * :param output: output tensor, shape is [C_out, oh, ow]
 * **/
template<typename T>
void conv2d_naive(const Tensor<T> &input, 
            const Tensor<T> &weight, 
            Tensor<T> &output, 
            const int stride_x,
            const int stride_y, 
            const int padding_x,
            const int padding_y, 
            const int groups) {

    // assert the dimension of input must be 3
    assert(input.shape().dim == 3 && "The dimension of input must be 3 for 2d-convolution");
    
    // T* input_data = input.data();
    // T* kernel_data = weight.data();
    // T* output_data = output.data();

    Shape input_shape = input.shape();
    Shape weight_shape = weight.shape();

    int in_channels = input_shape[0];
    int ih = input_shape[1];
    int iw = input_shape[2];

    int num_kernel = weight_shape[0];
    int kh = weight_shape[1];
    int kw = weight_shape[2];

    int out_channels = output.shape()[0];

    if (groups > 1) {
        int in_channels_per_group = in_channels / groups;
        int out_channels_per_group = out_channels / groups;
        int num_kernel_per_group = num_kernel / groups;

        for(int group = 0; group < groups; group++) {
            Tensor<T> group_input = input.sub(group * in_channels_per_group, (group + 1) * in_channels_per_group);
            Tensor<T> group_weight = weight.sub(group * num_kernel_per_group, (group + 1) * num_kernel_per_group);
            Tensor<T> group_output = output.sub(group * out_channels_per_group, (group + 1) * out_channels_per_group);
            conv2d_naive(group_input, group_weight, group_output, stride_x, stride_y, padding_x, padding_y, 1);
        }
    }else {
        for(int i = 0; i < out_channels; i++) {
            Tensor<T> subOutput = output.sub(i);
            for (int j = 0; j < in_channels; j++) {
                Tensor<T> subInput = input.sub(j);
                Tensor<T> subWeight = weight.sub(i).sub(j);
                conv2d_naive_single(subInput, subWeight, subOutput, stride_x, stride_y, padding_x, padding_y);
            }
        }
    }
}
    
} // namespace ops

} // namespace nninfer