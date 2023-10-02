#include <stdlib.h>

#ifndef MATH_H
#define MATH_H

namespace math
{

/**
 * \brief Generate a random integer in range [min, max)
 * **/
inline int randint(int min, int max)
{
    return rand() % (max - min) + min;
}
    
/**
 * \brief Generate a random float in range [0, 1)
 * **/
inline float random(int min, int max)
{
    float r = (float)rand() / RAND_MAX;
    return r * (max - min) + min;
}
    
} // namespace math

#endif // MATH_H