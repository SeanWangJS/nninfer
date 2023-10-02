#include "base_layer.h"
#include "ops/convolution.h"

namespace nninfer {

namespace layer
{

template <typename T>
class Convolution2D : public BaseLayer<T>
{
};    

} // namespace layer

} // namespace nninfer