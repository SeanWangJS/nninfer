#include "base_layer.h"

namespace nninfer
{

namespace layer
{

template <typename T>
class BatchNorm2d : public BaseLayer<T>
{

    public:
        
        Tensor<T> running_mean;
        Tensor<T> running_var;
        Tensor<T> weight;
        Tensor<T> bias;
        float eps;

        BatchNorm2d(Tensor<T> running_mean,
                    Tensor<T> running_var,
                    Tensor<T> weight,
                    Tensor<T> bias,
                    float eps);

        void forward(const Tensor<T> &input, 
                     Tensor<T> &output) override;

        Tensor<T> forward(const Tensor<T> &input) override;

};
    
} // namespace layer

    
} // namespace nninfer
