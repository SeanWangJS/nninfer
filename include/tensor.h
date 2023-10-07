#include "shape.h"

#ifndef TENSOR_H
#define TENSOR_H

namespace nninfer {

namespace tensor {

    template<typename T>
    class Tensor{
    private:
        T* mem;
        Shape _shape;

    public:
        // constructor
        Tensor();

        explicit Tensor(T* mem, Shape shape);
        
        T* data();
        
        T* data() const;

        Tensor<T> reshape(Shape shape);

        Shape shape();

        Shape shape() const;

        /**
         * \brief Get a sub tensor from a tensor
         * @param i the index of the sub tensor
         * **/
        Tensor<T> sub(int i) const;

        Tensor<T> sub(int i);

        /**
         * \brief Get a sub tensor from a tensor
         * @param start the start index of the sub tensor
         * @param end the end index of the sub tensor, not included
         * **/
        Tensor<T> sub(int start, int end) const;

        Tensor<T> sub(int start, int end);

        /**
         * \brief Add a scalar value
         * **/
        void addScalar(T scalar);

        void addScalar(T scalar) const;

        /**
         * \brief Multiple a scalar value
         * **/
        void mulScalar(T scalar);

        void mulScalar(T scalar) const;

        static Tensor<T> random(Shape shape, int min = 0, int max = 1);

        static Tensor<T> zeros(Shape shape);

        static Tensor<T> ones(Shape shape);

        static Tensor<T> arange(size_t start, size_t end, int step = 1);

    };

    template<typename T>
    void printMatrix(const T* data, const int row, const int col, const int start, const std::string indent) {
        std::cout << indent << "[";
        for (int i = 0; i < row; i++) {
            std::cout << "[";
            for (int j = 0; j < col - 1; j++) {
                // std::cout << "index: " << start + i * col + j << std::endl;
                std::cout << data[start + i * col + j] << ", ";
            }
            std::cout << data[start + i * col + col - 1] << "]";
            if(i < row - 1) { // not last row
                std::cout << std::endl;
                std::cout << indent;
            }
            
        }
        std::cout << "]" << std::endl;
    }

    template<typename T>
    void printTensor(const T* data, const int* shape, int dim, int start, std::string indent) {
        
        if(dim == 2) {
            printMatrix<T>(data, shape[0], shape[1], start, indent);
            return;
        }

        int subDim = dim - 1;
        int* subShape = new int[subDim];
        for(int i = 0; i < subDim; i++) {
            subShape[i] = shape[i + 1];
        }

        int firstDim = shape[0];
        int stride = 1;
        for(int i = 0; i < subDim; i++) {
            stride *= subShape[i];
        }

        for(int i = 0; i < firstDim; i++) {
            // std::cout << "[";
            printTensor<T>(data, subShape, subDim, start + i * stride, indent + "");
            std::cout << std::endl;
        }

    }
    template<typename T>
    inline std::ostream& operator<<(std::ostream &os, 
                             const Tensor<T> &tensor);

    /**
     * \brief Print and format the data of a tensor
     * for 1d tensor, print like 
     *      [1, 2, 3, 4]
     * for 2d tensor, print like 
     *      [[1, 2, 3], 
     *       [4, 5, 6]]
     * for 3d tensor, print like
     *      [[[1, 2, 3], 
     *        [4, 5, 6]],
     * 
     *       [[7, 8, 9],
     *        [10, 11, 12]]]
     * 
     */
    template<typename T>
    inline std::ostream& operator<<(std::ostream &os, 
                             const Tensor<T> &tensor) {
        const Shape shape = tensor.shape();
        int dim = shape.dim;
        size_t size = shape.size;
        T* data = tensor.data();
        if(dim == 1) {
            os << "[";
            for(int i = 0; i < size - 1; i++) {
                os << data[i] << ", ";
            }
            os << data[size - 1] << "]";
            return os;
        }
        printTensor<T>(data, shape.shape, dim, 0, "");
        return os;
    }

    // /**
    //  * \brief Add a scalar number to a tensor
    //  * 
    //  **/
    // template<typename T>
    // Tensor<T> operator+(const Tensor<T> &tensor, const T &num) {
    //     Shape shape = tensor.shape();
    //     size_t size = shape.size;
    //     T* data = tensor.data();
    //     T* newData = new T[size];
    //     for(int i = 0; i < size; i++) {
    //         newData[i] = data[i] + num;
    //     }
    //     return Tensor<T>(newData, shape);
    // }


} // namespace tensor

} // namespace nninfer

#endif // TENSOR_H