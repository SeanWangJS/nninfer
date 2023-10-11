#include "base_layer.h"

namespace nninfer
{

namespace layer
{

template <typename T>
class MaxPool2d : public BaseLayer<T> 
{

    public:

        std::pair<int, int> kernel_size;
        std::pair<int, int> stride;
        std::pair<int, int> padding;
        std::pair<int, int> dilation;
        bool ceil_mode;

        MaxPool2d(std::variant<std::pair<int,int>, int> kernel_size,
                  std::variant<std::pair<int,int>, int> stride,
                  std::variant<std::pair<int,int>, int> padding,
                  std::variant<std::pair<int,int>, int> dilation,
                  bool ceil_mode);

        void forward(const Tensor<T> &input, 
                     Tensor<T> &output) override;

        Tensor<T> forward(const Tensor<T> &input) override;

};
    
} // namespace layer

    
} // namespace nninfer
