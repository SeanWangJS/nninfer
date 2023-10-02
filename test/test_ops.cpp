#include "ops/convolution.h"
#include <gtest/gtest.h>

using namespace nninfer::ops;

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
// Test that the function returns the correct value for a valid index
TEST(GetPaddedDataTest, ValidIndex) {
    int data[] = {1, 2, 3, 4, 5, 6};
    int x = 1;
    int y = 1;
    int w = 3;
    int h = 2;
    int padding_x = 1;
    int padding_y = 1;
    int expected = 1;
    int result = get_padded_data(data, x, y, w, h, padding_x, padding_y);
    EXPECT_EQ(result, expected);
}

// Test that the function returns 0 for an index outside the bounds of the data
TEST(GetPaddedDataTest, InvalidIndex) {
    int data[] = {1, 2, 3, 4, 5, 6};
    int x = 3;
    int y = 4;
    int w = 3;
    int h = 2;
    int padding_x = 1;
    int padding_y = 1;
    int expected = 0;
    int result = get_padded_data(data, x, y, w, h, padding_x, padding_y);
    EXPECT_EQ(result, expected);
}

// Test that the function returns 0 for an index outside the bounds of the padded data
TEST(GetPaddedDataTest, InvalidPaddedIndex) {
    int data[] = {1, 2, 3, 4, 5, 6};
    int x = 0;
    int y = 0;
    int w = 3;
    int h = 2;
    int padding_x = 1;
    int padding_y = 1;
    int expected = 0;
    int result = get_padded_data(data, x, y, w, h, padding_x, padding_y);
    EXPECT_EQ(result, expected);
}

// Test that the function returns the correct value for a valid index with no padding
TEST(GetPaddedDataTest, NoPadding) {
    int data[] = {1, 2, 3, 4, 5, 6};
    int x = 1;
    int y = 1;
    int w = 3;
    int h = 2;
    int padding_x = 0;
    int padding_y = 0;
    int expected = 5;
    int result = get_padded_data(data, x, y, w, h, padding_x, padding_y);
    EXPECT_EQ(result, expected);
}

// Test that the function produces the correct output for a simple input and kernel
TEST(Conv2dNaiveSingleTest, SimpleInput) {
    float input_data[] = {1, 2, 3,
                       4, 5, 6,
                       7, 8, 9};
    float kernel_data[] = {1, 2,
                        3, 4};
    
    int stride_x = 1;
    int stride_y = 1;
    int padding_x = 0;
    int padding_y = 0;
    int use_bias = 0;
    Shape input_shape = Shape({3, 3});
    Shape kernel_shape = Shape({2, 2});
    Shape output_shape = Shape({2, 2});

    Tensor<float> input(input_data, input_shape);
    Tensor<float> kernel(kernel_data, kernel_shape);
    Tensor<float> output = Tensor<float>::zeros(output_shape);

    conv2d_naive_single(input, kernel, output, stride_x, stride_y, padding_x, padding_y, use_bias);
    float expect[] = {37, 47,
                      67, 77};
    float* actual = output.data();
    for(int i = 0; i < output_shape.size; i++) {
        EXPECT_EQ(actual[i], expect[i]);
    }
}

// Test that the function produces the correct output for a larger input and kernel
TEST(Conv2dNaiveSingleTest, LargeInput) {
    Shape input_shape = Shape({5, 5});
    Shape kernel_shape = Shape({3, 3});
    Shape output_shape = Shape({3, 3});
    float input_data[] = {1, 2, 3, 4, 5, 
                          6, 7, 8, 9, 10, 
                          11, 12, 13, 14, 15, 
                          16, 17, 18, 19, 20, 
                          21, 22, 23, 24, 25};
    float kernel_data[] = {1, 1, 1, 
                           1, 1, 1, 
                           1, 1, 1};
    int stride_x = 1;
    int stride_y = 1;
    int padding_x = 0;
    int padding_y = 0;
    int use_bias = 0;

    Tensor<float> input(input_data, input_shape);
    Tensor<float> kernel(kernel_data, kernel_shape);
    Tensor<float> output = Tensor<float>::zeros(output_shape);
    conv2d_naive_single(input, kernel, output, stride_x, stride_y, padding_x, padding_y, use_bias);
    
    EXPECT_EQ(output.data()[0], 63);
    EXPECT_EQ(output.data()[8], 171);

}

// Test that the function produces the correct output when using stride > 1
TEST(Conv2dNaiveSingleTest, Stride) {

    float input_data[] = {1, 2, 3, 4,
                          5, 6, 7, 8, 
                          9, 10, 11, 12, 
                          13, 14, 15, 16};
    float kernel_data[] = {1, 1,
                           1, 1};
    
    int stride_x = 2;
    int stride_y = 2;
    int padding_x = 0;
    int padding_y = 0;
    int use_bias = 0;
    Shape input_shape = Shape({4, 4});
    Shape kernel_shape = Shape({2, 2});
    Shape output_shape = Shape({2, 2});

    Tensor<float> input(input_data, input_shape);
    Tensor<float> kernel(kernel_data, kernel_shape);
    Tensor<float> output = Tensor<float>::zeros(output_shape);

    conv2d_naive_single(input, kernel, output, stride_x, stride_y, padding_x, padding_y, use_bias);
    float expect[] = {14, 22,
                      46, 54};
    float* actual = output.data();
    for(int i = 0; i < output_shape.size; i++) {
        EXPECT_EQ(actual[i], expect[i]);
    }
}

// Test that the function produces the correct output when using padding > 0
TEST(Conv2NavieSingleTest, Padding) {
    float input_data[] = {1, 2, 3,
                          4, 5, 6,
                          7, 8, 9};
    float kernel_data[] = {1, 1, 1,
                           1, 1, 1,
                           1, 1, 1};
    
    int stride_x = 1;
    int stride_y = 1;
    int padding_x = 1;
    int padding_y = 1;
    int use_bias = 0;
    Shape input_shape = Shape({3, 3});
    Shape kernel_shape = Shape({3, 3});
    Shape output_shape = Shape({3, 3});

    Tensor<float> input(input_data, input_shape);
    Tensor<float> kernel(kernel_data, kernel_shape);
    Tensor<float> output = Tensor<float>::zeros(output_shape);

    conv2d_naive_single(input, kernel, output, stride_x, stride_y, padding_x, padding_y, use_bias);
    float expect[] = {12, 21, 16,
                      27, 45, 33,
                      24, 39, 28};
    float* actual = output.data();
    for(int i = 0; i < output_shape.size; i++) {
        EXPECT_EQ(actual[i], expect[i]);
    }
}