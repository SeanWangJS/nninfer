#include <assert.h>

#include "tensor.h"
#include "shape.h"
#include "ops/utils.h"

#pragma once

using namespace nninfer::tensor;

namespace nninfer
{

namespace ops
{

template <typename T>
void max_pool2d(const Tensor<T> &input, 
                Tensor<T> &output,
                const int kernel_x,
                const int kernel_y,
                const int stride_x,
                const int stride_y,
                const int padding_x,
                const int padding_y) {
    
    assert(input.shape().dim == 4 && "The dimension of input must be 4 for batch 2d-pooling");
    assert(output.shape().dim == 4 && "The dimension of output must be 4 for batch 2d-pooling");

    int batch_size = input.shape()[0];
    for(int b = 0; b < batch_size; b++) {
        Tensor<T> subInput = input.sub(b);
        Tensor<T> subOutput = output.sub(b);
        int num_channels = input.shape()[1];

        for(int c = 0; c < num_channels; c++) {
            Tensor<T> channel_input = subInput.sub(c);
            Tensor<T> channel_output = subOutput.sub(c);
            T* input_data = channel_input.data();
            T* output_data = channel_output.data();

            int ih = channel_input.shape()[0];
            int iw = channel_input.shape()[1];
            int oh = channel_output.shape()[0];
            int ow = channel_output.shape()[1];
            
            for(int i = 0; i < oh; i++) {
                for(int j = 0; j < ow; j++) {

                    int y = i * stride_y;
                    int x = j * stride_x;
                    T max = 0;
                    for(int ki = 0; ki < kernel_y; ki++) {
                        for(int kj = 0; kj < kernel_x; kj++) {
                            T data = get_padded_data(input_data, x + kj, y + ki, iw, ih, padding_x, padding_y);
                            if(data > max) {
                                max = data;
                            }
                        }
                    }
                    output_data[j + i * ow] = max;                    

                }
            }


        }
    }

}

}
    
} // namespace nninfer
