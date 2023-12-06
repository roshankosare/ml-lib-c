#include <math.h>
#include <stdlib.h>
#ifndef NN_UTILS_H
#define NN_UTILS_H

typedef enum
{
    ReLU,
    Sigmoid,
    Linear
} AF;
#define RELU_PARAM 0.001
#define ARRAY_LEN(x) sizeof((x)) / sizeof((x)[0])
float rand_float();
float sigmoidf(float x);
float sigmoidf_derivative(float x);
float ReLu(float x);
float ReLu_derivative(float x);

float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.0 / (1.0 + expf(-x));
}

float sigmoidf_derivative(float x)
{
    return x * (1 - x);
}

float ReLu(float x)
{
    return x > 0 ? x : RELU_PARAM * x;
}
float ReLu_derivative(float x)
{
    return x > 0 ? 1.0 : RELU_PARAM;
}
float act(float x, AF af)
{
    switch (af)
    {
    case ReLU:
        return ReLu(x);
        break;
    case Sigmoid:
        return sigmoidf(x);
        break;
    case Linear:
        return x;
    default:
        break;
    }
}

float dact(float x, AF af)
{

    switch (af)
    {
    case Sigmoid:
        return sigmoidf_derivative(x);
        break;
    case ReLU:
        return ReLu_derivative(x);
        break;
    case Linear:
        return x;

    default:
        break;
    }
}

#endif