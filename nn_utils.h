#include <math.h>
#include <stdlib.h>
#ifndef NN_UTILS_H
#define NN_UTILS_H

typedef enum
{
    ReLU,
    Sigmoid
} AF;
#define RELU_PARAM 0.1f
#define ARRAY_LEN(x) sizeof((x)) / sizeof((x)[0])
float rand_float();
float sigmoidf(float x);
float ReLu(float x);

float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float ReLu(float x)
{
    return x > 0 ? x : 0;
}

float dact(float x, AF af)
{

    switch (af)
    {
    case Sigmoid:
        return x * (1 - x);
        break;
    case ReLU:
        return x >= 0 ? 1 : 0;
        break;

    default:
        break;
    }
}

#endif