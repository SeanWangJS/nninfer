#include <iostream>

#include <gtest/gtest.h>

#include "ops/convolution.h"
#include "ops/batch_norm.h"

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
    Shape input_shape = Shape({3, 3});
    Shape kernel_shape = Shape({2, 2});
    Shape output_shape = Shape({2, 2});

    Tensor<float> input(input_data, input_shape);
    Tensor<float> kernel(kernel_data, kernel_shape);
    Tensor<float> output = Tensor<float>::zeros(output_shape);

    conv2d_naive_single(input, kernel, output, stride_x, stride_y, padding_x, padding_y);
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

    Tensor<float> input(input_data, input_shape);
    Tensor<float> kernel(kernel_data, kernel_shape);
    Tensor<float> output = Tensor<float>::zeros(output_shape);
    conv2d_naive_single(input, kernel, output, stride_x, stride_y, padding_x, padding_y);
    
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
    Shape input_shape = Shape({4, 4});
    Shape kernel_shape = Shape({2, 2});
    Shape output_shape = Shape({2, 2});

    Tensor<float> input(input_data, input_shape);
    Tensor<float> kernel(kernel_data, kernel_shape);
    Tensor<float> output = Tensor<float>::zeros(output_shape);

    conv2d_naive_single(input, kernel, output, stride_x, stride_y, padding_x, padding_y);
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
    Shape input_shape = Shape({3, 3});
    Shape kernel_shape = Shape({3, 3});
    Shape output_shape = Shape({3, 3});

    Tensor<float> input(input_data, input_shape);
    Tensor<float> kernel(kernel_data, kernel_shape);
    Tensor<float> output = Tensor<float>::zeros(output_shape);

    conv2d_naive_single(input, kernel, output, stride_x, stride_y, padding_x, padding_y);
    float expect[] = {12, 21, 16,
                      27, 45, 33,
                      24, 39, 28};
    float* actual = output.data();
    for(int i = 0; i < output_shape.size; i++) {
        EXPECT_EQ(actual[i], expect[i]);
    }
}

// Test that the function produces the correct output for a simple input and kernel
TEST(Conv2dNaiveTest, SimpleInput) {
    float input_data[] = {1, 2, 3, 
                          4, 5, 6, 
                          7, 8, 9};
    float kernel_data[] = {1, 2, 
                           3, 4};                          
    Shape input_shape = Shape({1, 3, 3});
    Shape kernel_shape = Shape({1, 1, 2, 2});
    Shape output_shape = Shape({1, 2, 2});
    int stride_x = 1;
    int stride_y = 1;
    int padding_x = 0;
    int padding_y = 0;
    int groups = 1;
    Tensor<float> input(input_data, input_shape);
    Tensor<float> kernel(kernel_data, kernel_shape);
    Tensor<float> output = Tensor<float>::zeros(output_shape);

    conv2d_naive(input, kernel, output, stride_x, stride_y, padding_x, padding_y, groups);
    float expect[] = {37, 47, 
                      67, 77};
    float* actual = output.data();
    for(int i = 0; i < output_shape.size; i++) {
        EXPECT_EQ(actual[i], expect[i]);
    }
}

// Test that the function produces the correct output for multiple channels
TEST(Conv2dNaiveTest, MultipleChannels) {
    Shape input_shape = Shape({2, 3, 3});
    Shape output_shape = Shape({3, 2, 2});
    Shape kernel_shape = Shape({3, 2, 2, 2});
    float input_data[] = {1, 2, 3, 
                        4, 5, 6, 
                        7, 8, 9, 
                        
                        10, 11, 12,
                        13, 14, 15,
                        16, 17, 18};
    float kernel_data[] = {1, 1, 1, 1,
                           1, 1, 1, 1,
                           
                           1, 1, 1, 1,
                           1, 1, 1, 1,
                           
                           1, 1, 1, 1,
                           1, 1, 1, 1};

    Tensor<float> input(input_data, input_shape);
    Tensor<float> kernel(kernel_data, kernel_shape);
    Tensor<float> output = Tensor<float>::zeros(output_shape);                           
    int stride_x = 1;
    int stride_y = 1;
    int padding_x = 0;
    int padding_y = 0;
    int groups = 1;
    conv2d_naive(input, kernel, output, stride_x, stride_y, padding_x, padding_y, groups);
    EXPECT_EQ(output.sub(0).data()[0], 60); // first element
    EXPECT_EQ(output.sub(2).data()[3], 92); // last element
}

// Test that the function produces the correct output for multiple groups
TEST(Conv2dNaiveTest, MultipleGroups) {

    int stride_x = 1;
    int stride_y = 1;
    int padding_x = 0;
    int padding_y = 0;
    int groups = 2;
    Shape input_shape = Shape({4, 3, 3});
    Shape output_shape = Shape({6, 2, 2});
    Shape kernel_shape = Shape({6, 4 / groups, 2, 2});

    Tensor<float> input = Tensor<float>::arange(1, input_shape.size + 1, 1).reshape(input_shape);
    // std::cout << input << std::endl;
    Tensor<float> weight = Tensor<float>::ones(kernel_shape);
    // std::cout << weight << std::endl;
    Tensor<float> output = Tensor<float>::zeros(output_shape);
    conv2d_naive(input, weight, output, stride_x, stride_y, padding_x, padding_y, groups);

    // std::cout << output << std::endl;
    EXPECT_EQ(output.sub(0).data()[0], 60); // first element
    EXPECT_EQ(output.sub(5).data()[3], 236); // last element

}

TEST(BatchNormTest, Test1) {
    // Test batch_norm with a single channel
    float data[] = {1, 2, 3, 4};
    float running_mean[] = {2};
    float running_var[] = {1};
    float weight[] = {1};
    float bias[] = {0};
    float expected[] = {-1, 0, 1, 2};
    float eps = 1e-5f;
    Tensor<float> data_tensor(data, {1, 1, 2, 2});
    Tensor<float> running_mean_tensor(running_mean, {1});
    Tensor<float> running_var_tensor(running_var, {1});
    Tensor<float> weight_tensor(weight, {1});
    Tensor<float> bias_tensor(bias, {1});
    batch_norm(data_tensor, running_mean_tensor, running_var_tensor, weight_tensor, bias_tensor, eps);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(data[i], expected[i], 0.001);
    }
}

TEST(BatchNormTest, Test2) {
    // Test batch_norm with multiple channels
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float running_mean[] = {2, 4};
    float running_var[] = {1, 2};
    float weight[] = {1, 2};
    float bias[] = {0, 1};
    float expected[] = {-1, 0, 1, 2, 2.4142f, 3.8284f, 5.2426f, 6.6568f};
    float eps = 1e-5f;
    Tensor<float> data_tensor(data, {1, 2, 2, 2});
    Tensor<float> running_mean_tensor(running_mean, {2});
    Tensor<float> running_var_tensor(running_var, {2});
    Tensor<float> weight_tensor(weight, {2});
    Tensor<float> bias_tensor(bias, {2});
    batch_norm(data_tensor, running_mean_tensor, running_var_tensor, weight_tensor, bias_tensor, eps);
    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(data[i], expected[i], 0.001);
    }
}

TEST(BatchNormTest, Test3) {
    // Test batch_norm with batch size > 1
    Shape data_shape = Shape({4, 2, 3, 3});
    Shape running_mean_shape = Shape({2});
    Shape running_var_shape = Shape({2});
    Shape weight_shape = Shape({2});
    Shape bias_shape = Shape({2});

    Tensor<float> data = Tensor<float>::arange(1, data_shape.size + 1, 1).reshape(data_shape);
    float running_mean[] = {2, 4};
    float running_var[] = {1, 2};
    float weight[] = {1, 2};
    float bias[] = {0, 1};
    float eps = 1e-5f;
    Tensor<float> running_mean_tensor(running_mean, running_mean_shape);
    Tensor<float> running_var_tensor(running_var, running_var_shape);
    Tensor<float> weight_tensor(weight, weight_shape);
    Tensor<float> bias_tensor(bias, bias_shape);
    batch_norm(data, running_mean_tensor, running_var_tensor, weight_tensor, bias_tensor, eps);

    ASSERT_NEAR(data.sub(0).sub(0).data()[0], -1, 0.001);
    ASSERT_NEAR(data.sub(3).sub(1).data()[8], 97.1663, 0.001);
}