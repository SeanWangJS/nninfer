#include "tensor.h"

using namespace nninfer::tensor;

namespace nninfer
{

namespace ops
{

/**
 * \brief Inplace batch normalization
 * for each channel, perform
 *     y = (x - mean) / sqrt(var + eps) * weight + bias
 * @param data Tensor, shape: [N, C, H, W]
 * @param running_mean Running mean, shape: [C]
 * @param running_var Running variance, shape: [C]
 * @param weight Weight, shape: [C]
 * @param bias Bias, shape: [C]
 * @param eps Epsilon
 * **/
template <typename T>
void batch_norm(const Tensor<T> &data,
                const Tensor<T> &running_mean,
                const Tensor<T> &running_var,
                const Tensor<T> &weight,
                const Tensor<T> &bias,
                const float eps) {

    Shape data_shape = data.shape();

    int batch_size = data_shape.shape[0];
    int n_channels = data_shape.shape[1];
    int height = data_shape.shape[2];
    int width = data_shape.shape[3];

    for(int b = 0; b < batch_size; b++) {
        Tensor<T> subData = data.sub(b);
        for(int c = 0; c < n_channels; c++) {
            T mean = running_mean.data()[c];
            T var = running_var.data()[c];
            T w = weight.data()[c];
            T bi = bias.data()[c];
            Tensor<T> channelData = subData.sub(c);
            channelData.addScalar(-mean);
            channelData.mulScalar(1 / sqrt(var + eps));
            channelData.mulScalar(w);
            channelData.addScalar(bi);
        }
    }
}
    
} // namespace ops

    
} // namespace nninfer


