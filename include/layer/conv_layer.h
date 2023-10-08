#include <variant>

#include "base_layer.h"

namespace nninfer 
{

namespace layer
{

template <typename T>
class Conv2d : public BaseLayer<T>
{
    public:
        int in_channels;
        int out_channels;
        std::pair<int, int> kernel_size;
        std::pair<int, int> stride;
        std::pair<int, int> padding;
        int groups;
        bool use_bias;

        Tensor<T> weight;
        Tensor<T> bias;

        Conv2d(int in_channels,
               int out_channels,
               std::variant<std::pair<int,int>, int> kernel_size,
               std::variant<std::pair<int,int>, int> stride = 1,
               std::variant<std::pair<int,int>, int> padding = 0,
               int groups = 1,
               bool use_bias = true);

        void forward(const Tensor<T> &input, 
                     Tensor<T> &output) override;

        Tensor<T> forward(const Tensor<T> &input) override;

};    

} // namespace layer

} // namespace nninfer