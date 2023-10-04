#include <gtest/gtest.h>

#include "tensor.h"
#include "layer/conv_layer.h"

using namespace nninfer::layer;
using namespace nninfer::tensor;

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(ConvLayer, ConvLayerConstructor) {

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