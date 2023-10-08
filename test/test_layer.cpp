#include <gtest/gtest.h>

#include "tensor.h"
#include "layer/bn_layer.h"
#include "layer/conv_layer.h"

using namespace nninfer::layer;
// using namespace nninfer::tensor;

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(ConvLayer, Constructor) {

    int in_channels = 2;
    int out_channels = 3;
    int kernel_size = 3;
    int stride = 1;
    int padding = 0;
    int groups = 1;
    bool use_bias = false;

    Conv2d<float> conv2d_layer(in_channels, out_channels, kernel_size, stride, padding, groups, use_bias);

    EXPECT_EQ(conv2d_layer.in_channels, in_channels);
    EXPECT_EQ(conv2d_layer.out_channels, out_channels);
    EXPECT_EQ(conv2d_layer.kernel_size.first, kernel_size);
    EXPECT_EQ(conv2d_layer.kernel_size.second, kernel_size);
    EXPECT_EQ(conv2d_layer.stride.first, stride);
    EXPECT_EQ(conv2d_layer.stride.second, stride);
    EXPECT_EQ(conv2d_layer.padding.first, padding);
    EXPECT_EQ(conv2d_layer.padding.second, padding);
    EXPECT_EQ(conv2d_layer.groups, groups);
    EXPECT_EQ(conv2d_layer.use_bias, use_bias);

}

TEST(ConvLayer, Forward) {

    int in_channels = 4;
    int out_channels = 6;
    int kernel_size = 2;
    int stride = 1;
    int padding = 0;
    int groups = 2;
    bool use_bias = false;

    Conv2d<float> conv2d_layer(in_channels, out_channels, kernel_size, stride, padding, groups, use_bias);

    // customize weight
    Shape weight_shape = conv2d_layer.weight.shape();
    Tensor<float> weight = Tensor<float>::ones(weight_shape);
    conv2d_layer.weight = weight;

    // construct input tensor
    Shape input_shape = Shape({1, in_channels, 3, 3});
    Tensor<float> input = Tensor<float>::arange(1, input_shape.size + 1, 1).reshape(input_shape);

    // construct output tensor
    Shape output_shape = Shape({1, out_channels, 2, 2});
    Tensor<float> output = Tensor<float>::zeros(output_shape);

    conv2d_layer.forward(input, output);

    EXPECT_EQ(output.sub(0).sub(0).data()[0], 60); // first element
    EXPECT_EQ(output.sub(0).sub(5).data()[3], 236); // last element
}

TEST(ConvLayer, ForwardOutput) {

    int in_channels = 4;
    int out_channels = 6;
    int kernel_size = 2;
    int stride = 1;
    int padding = 0;
    int groups = 2;
    bool use_bias = false;

    Conv2d<float> conv2d_layer(in_channels, out_channels, kernel_size, stride, padding, groups, use_bias);

    // customize weight
    Shape weight_shape = conv2d_layer.weight.shape();
    Tensor<float> weight = Tensor<float>::ones(weight_shape);
    conv2d_layer.weight = weight;

    // construct input tensor
    Shape input_shape = Shape({1, in_channels, 3, 3});
    Tensor<float> input = Tensor<float>::arange(1, input_shape.size + 1, 1).reshape(input_shape);

    Tensor<float> output = conv2d_layer.forward(input);
    EXPECT_EQ(output.sub(0).sub(0).data()[0], 60); // first element
    EXPECT_EQ(output.sub(0).sub(5).data()[3], 236); // last element

}

TEST(TestBatchNormLayer, Constructor) {

    // Shape data_shape = Shape({4, 2, 3, 3});
    // Tensor<float> data = Tensor<float>::arange(1, data_shape.size + 1, 1).reshape(data_shape);

    float running_mean[] = {2, 4};
    float running_var[] = {1, 2};
    float weight[] = {1, 2};
    float bias[] = {0, 1};
    float eps = 1e-5f;
    Tensor<float> running_mean_tensor(running_mean, {2, 4});
    Tensor<float> running_var_tensor(running_var, {1, 2});
    Tensor<float> weight_tensor(weight, {1, 2});
    Tensor<float> bias_tensor(bias, {0, 1});

    BatchNorm2d<float> bn_layer(running_mean_tensor, running_var_tensor, weight_tensor, bias_tensor, eps);

    EXPECT_EQ(bn_layer.running_mean.data()[0], 2);
    EXPECT_EQ(bn_layer.running_mean.data()[1], 4);
    EXPECT_EQ(bn_layer.running_var.data()[0], 1);
    EXPECT_EQ(bn_layer.running_var.data()[1], 2);
    EXPECT_EQ(bn_layer.weight.data()[0], 1);
    EXPECT_EQ(bn_layer.weight.data()[1], 2);
    EXPECT_EQ(bn_layer.bias.data()[0], 0);
    EXPECT_EQ(bn_layer.bias.data()[1], 1);
    EXPECT_EQ(bn_layer.eps, eps);
}

TEST(TestBatchNormLayer, Forward) {

    Shape data_shape = Shape({4, 2, 3, 3});
    Tensor<float> data = Tensor<float>::arange(1, data_shape.size + 1, 1).reshape(data_shape);

    float running_mean[] = {2, 4};
    float running_var[] = {1, 2};
    float weight[] = {1, 2};
    float bias[] = {0, 1};
    float eps = 1e-5f;
    Tensor<float> running_mean_tensor(running_mean, {2, 4});
    Tensor<float> running_var_tensor(running_var, {1, 2});
    Tensor<float> weight_tensor(weight, {1, 2});
    Tensor<float> bias_tensor(bias, {0, 1});

    BatchNorm2d<float> bn_layer(running_mean_tensor, running_var_tensor, weight_tensor, bias_tensor, eps);

    Tensor<float> output = bn_layer.forward(data);

    ASSERT_NEAR(output.sub(0).sub(0).data()[0], -1, 0.001);
    ASSERT_NEAR(output.sub(3).sub(1).data()[8], 97.1663, 0.001);

}