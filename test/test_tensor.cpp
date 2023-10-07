// test with GTest
#include <iostream>
#include <gtest/gtest.h>

#include "tensor.h"

using namespace nninfer::tensor;

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(tensor_constructor, default_tensor_constructor) {

    // Test constructor with memory and shape
    int data[] = {1, 2, 3, 4, 5, 6};
    Shape shape = Shape({2, 3});

    Tensor<int> t2(data, shape);

    EXPECT_EQ(t2.data()[0], 1);
    EXPECT_EQ(t2.data()[1], 2);
    EXPECT_EQ(t2.data()[2], 3);
    EXPECT_EQ(t2.data()[3], 4);
    EXPECT_EQ(t2.data()[4], 5);
    EXPECT_EQ(t2.data()[5], 6);    
}

TEST(tensor_data, tensor_data) {
    int data[] = {1,2,3,4,5,6};
    Shape shape = Shape({2, 3});

    Tensor<int> t(data, shape);
    
    // Test non-const data()
    int* d = t.data();
    d[0] = 100;
    EXPECT_EQ(d[0], 100);

    // Test const data()
    const Tensor<int> ct(data, shape);
    const int* cd = ct.data();
    EXPECT_EQ(cd[0], 100); 
}

TEST(tensor_reshape, shape) {
    int data[] = {1,2,3,4,5,6};
    Shape shape = Shape({2, 3});

    Tensor<int> t(data, shape);
    
    
}

TEST(subtensor, create_subtensor) {
    int data[] = {1,2,3,4,5,6};
    Shape shape = Shape({2, 3});

    Tensor<int> t(data, shape);
    Tensor<int> t2 = t.sub(0);
    EXPECT_EQ(t2.data()[0], 1);
    EXPECT_EQ(t2.data()[1], 2);
    EXPECT_EQ(t2.data()[2], 3);
    Tensor<int> t3 = t.sub(1);
    EXPECT_EQ(t3.data()[0], 4);
    EXPECT_EQ(t3.data()[1], 5);
    EXPECT_EQ(t3.data()[2], 6);
}

TEST(subtensor, index_out_of_range_error) {

    int data[] = {1,2,3,4,5,6};
    Shape shape = Shape({2, 3});

    Tensor<int> t(data, shape);
    EXPECT_THROW(t.sub(2), std::invalid_argument);

}

TEST(subtensor, sub_a_scalar_error) {
    int data[] = {1};
    Shape shape = Shape({1});

    Tensor<int> t(data, shape);
    EXPECT_THROW(t.sub(0), std::invalid_argument);
}

TEST(arange, normal_input) {
    Tensor<int> t = Tensor<int>::arange(0, 10, 1);
    EXPECT_EQ(t.data()[0], 0);
    EXPECT_EQ(t.data()[9], 9);
}

TEST(arange, undividable_input) {
    Tensor<int> t = Tensor<int>::arange(0, 10, 3);
    EXPECT_EQ(t.shape().size, 4);
    EXPECT_EQ(t.data()[0], 0);
    EXPECT_EQ(t.data()[1], 3);
    EXPECT_EQ(t.data()[2], 6);
    EXPECT_EQ(t.data()[3], 9);
}

TEST(sub_tensor, sub_range_tensor) {

    Shape shape = Shape({4, 5});
    Tensor<int> t = Tensor<int>::arange(1, 20, 1).reshape(shape);

    Tensor<int> subT = t.sub(1, 3);

    EXPECT_EQ(subT.shape().size, 10);
    EXPECT_EQ(subT.data()[0], 6);
    EXPECT_EQ(subT.data()[9], 15);
}

TEST(sub_tensor, sub_tensor_bound_case) {
    Shape shape = Shape({4, 5});
    Tensor<int> t = Tensor<int>::arange(1, 20, 1).reshape(shape);

    Tensor<int> subT = t.sub(1, 2);

    EXPECT_EQ(subT.shape().size, 5);
    EXPECT_EQ(subT.data()[0], 6);
    EXPECT_EQ(subT.data()[4], 10);
}

TEST(Add, AddScalar) {
    Shape shape = Shape({4, 5});
    const Tensor<int> t = Tensor<int>::arange(0, 20, 1).reshape(shape);

    t.addScalar(10);

    EXPECT_EQ(t.data()[0], 10);
    EXPECT_EQ(t.data()[19], 29);
}

TEST(Mul, MulScalar) {
    Shape shape = Shape({4, 5});
    const Tensor<int> t = Tensor<int>::arange(0, 20, 1).reshape(shape);

    t.mulScalar(10);

    EXPECT_EQ(t.data()[0], 0);
    EXPECT_EQ(t.data()[19], 190);
}