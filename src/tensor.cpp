#include <iostream>

#include "tensor.h"
#include "math_op.h"

namespace nninfer {
    
namespace tensor{

template class Tensor<int>;
template class Tensor<float>;

// template<typename T>
// static Tensor<T> Tensor<T>::random(Shape shape){

// }

template<typename T>
Tensor<T>::Tensor() = default;

template<typename T>
Tensor<T>::Tensor(T* mem, Shape shape){
    this->mem = mem;
    this->_shape = shape;
}

template<typename T>
T* Tensor<T>::data(){
    return mem;
}

template<typename T>
T* Tensor<T>::data() const{
    return mem;
}

template<typename T>
Shape Tensor<T>::shape(){
    return _shape;
}

template<typename T>
Shape Tensor<T>::shape() const{
    return _shape;
}

template<typename T>
Tensor<T> Tensor<T>::reshape(Shape shape){
    return Tensor(this->mem, shape);
}

template<typename T>
Tensor<T> Tensor<T>::sub(int i) {
    if (i >= _shape.shape[0]) {
        throw std::invalid_argument("Index out of range");
    }
    int subDim = _shape.dim - 1;
    if (subDim <= 0) {
        throw std::invalid_argument("Cannot sub a tensor with scalar");
    }
    int* s = new int[subDim];
    for (int j = 0; j < subDim; j++) {
        s[j] = _shape[j + 1];
    }
    Shape subShape(s, subDim);
    return Tensor(this->mem + i * subShape.size, subShape);

}

template<typename T>
Tensor<T> Tensor<T>::sub(int i) const {
    const Shape _shape = this->shape();
    if (i >= _shape.shape[0]) {
        throw std::invalid_argument("Index out of range");
    }
    int subDim = _shape.dim - 1;
    if (subDim <= 0) {
        throw std::invalid_argument("Cannot sub a tensor with scalar");
    }
    int* s = new int[subDim];
    for (int j = 0; j < subDim; j++) {
        s[j] = _shape[j + 1];
    }
    Shape subShape(s, subDim);
    return Tensor(this->mem + i * subShape.size, subShape);
}

template<>
Tensor<int> Tensor<int>::random(Shape shape, int min, int max){
    size_t size = shape.size;
    int* data = new int[size];
    for (int i = 0; i < size; i++) {
        data[i] = math::randint(min, max);
    }
    return Tensor(data, shape);
}

template<>
Tensor<float> Tensor<float>::random(Shape shape, int min, int max){
    size_t size = shape.size;
    float* data = new float[size];
    for (int i = 0; i < size; i++) {
        data[i] = math::random(min, max);
        
    }
    return Tensor(data, shape);
}

template <typename T>
Tensor<T> Tensor<T>::zeros(Shape shape){
    size_t size = shape.size * sizeof(T);
    T* data = (T*)malloc(size);

    for (int i = 0; i < shape.size; i++) {
        data[i] = static_cast<T>(0);
    }

    return Tensor(data, shape);
}

template <typename T>
Tensor<T> Tensor<T>::ones(Shape shape){
    size_t size = shape.size * sizeof(T);
    T* data = (T*)malloc(size);

    for (int i = 0; i < shape.size; i++) {
        data[i] = static_cast<T>(1);
    }

    return Tensor(data, shape);
}

} // namespace tensor
} // namespace nninfer
