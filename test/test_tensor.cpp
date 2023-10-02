// test with GTest
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
    int* s = new int[2];
    s[0] = 2;
    s[1] = 3;
    Shape shape(s, 2);

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
    int* s = new int[2];
    s[0] = 2;
    s[1] = 3;
    Shape shape(s, 2);

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
    int* s = new int[2];
    s[0] = 2;
    s[1] = 3;
    Shape shape(s, 2);

    Tensor<int> t(data, shape);
    
    
}

TEST(subtensor, create_subtensor) {
    int data[] = {1,2,3,4,5,6};
    int* s = new int[2];
    s[0] = 2;
    s[1] = 3;
    Shape shape(s, 2);

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
    int* s = new int[2];
    s[0] = 2;
    s[1] = 3;

    Shape shape(s, 2);

    Tensor<int> t(data, shape);
    EXPECT_THROW(t.sub(2), std::invalid_argument);

}

TEST(subtensor, sub_a_scalar_error) {
    int data[] = {1};
    int* s = new int[1];
    s[0] = 1;
    Shape shape(s, 1);

    Tensor<int> t(data, shape);
    EXPECT_THROW(t.sub(0), std::invalid_argument);
}