#include <gtest/gtest.h>

#include "tensor.h"
#include "layer/conv_layer.h"

using namespace nninfer::layer;
using namespace nninfer::tensor;

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