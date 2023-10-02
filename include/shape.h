#include <iostream>

#include <array>

#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H

namespace nninfer {

namespace tensor {


    struct Shape
    {
        int size;
        int* shape;
        int dim;

        Shape() = default;
        Shape(int* shape, int dim) {
            this->shape = shape;
            this->dim = dim;
            size = 1;
            for (int i = 0; i < dim; i++) {
                size *= shape[i];
            }
        }

        int operator[](int i) {
            return shape[i];
        }

    };

    
    inline std::ostream& operator<<(std::ostream &os, 
                             const Shape &shape) {
        os << "[";
        for (int i = 0; i < shape.dim - 1; i++) {
            os << shape.shape[i];
            os << ", ";
            
        }
        os << shape.shape[shape.dim - 1];
        os << "]";
        return os;
    }

} // namespace tensor

} // namespace nninfer

#endif // TENSOR_SHAPE_H