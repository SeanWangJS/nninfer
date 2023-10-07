#include "tensor.h"
#include "base_layer.h"

using namespace nninfer::tensor;

namespace nninfer
{

namespace layer
{

template <typename T>
class BatchNorm2dLayer : public BaseLayer<T>
{

    public:


        void forward(const Tensor<T> &input, 
                     Tensor<T> &output) override;

        Tensor<T> forward(const Tensor<T> &input) override;

}
    
} // namespace layer

    
} // namespace nninfer
