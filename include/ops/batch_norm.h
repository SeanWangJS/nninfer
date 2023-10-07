#include "tensor.h"

using namespace nninfer::tensor;

namespace nninfer
{

namespace ops
{

/**
 * \brief Batch normalization
 * for each channel, perform
 *     y = (x - mean) / sqrt(var + eps) * weight + bias
 * @param input Input tensor, shape: [N, C, H, W]
 * @param output Output tensor, shape: [N, C, H, W]
 * @param running_mean Running mean, shape: [C]
 * @param running_var Running variance, shape: [C]
 * @param weight Weight, shape: [C]
 * @param bias Bias, shape: [C]
 * @param eps Epsilon
 * **/
template <typename T>
void batch_norm(const Tensor<T> &input,
                Tensor<T> &output,
                const Tensor<T> &running_mean,
                const Tensor<T> &running_var,
                const Tensor<T> &weight,
                const Tensor<T> &bias,
                const float eps) {

    Shape input_shape = input.shape();

    int batch_size = input_shape.shape[0];
}
    
} // namespace ops

    
} // namespace nninfer


