#pragma once

template<typename T>
inline T get_padded_data(const T* data, 
           const int x,
           const int y,
           const int w,
           const int h,
           const int padding_x,
           const int padding_y) {

    int x_ = x - padding_x;
    int y_ = y - padding_y;
    if(x_ < 0 || y_ < 0 || x_ >= w || y_ >= h) {
        return static_cast<T>(0);
    }

    return data[x_ + y_ * w];
}