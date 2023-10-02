#include <assert.h>

#include "tensor.h"

using namespace nninfer::tensor;

namespace nninfer{
    
namespace ops{
    
template<typename T>
void conv2d_naive(const Tensor<T> &input, 
            const Tensor<T> &kernel, 
            Tensor<T> &output, 
            const int stride_x,
            const int stride_y, 
            const int padding_x,
            const int padding_y, 
            const int groups,
            const int use_bias) {

    // assert the dimension of input must be 3
    assert(input.shape().dim == 3, "The dimension of input must be 3 for 2d-convolution");
    
    T* input_data = input.data();
    T* kernel_data = kernel.data();
    T* output_data = output.data();


}
    
} // namespace ops

} // namespace nninfer