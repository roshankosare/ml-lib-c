#include <math.h>
#include <stdlib.h>
#ifndef NN_UTILS_H
#define NN_UTILS_H

typedef enum
{
    RELU,
    SIG,
    LINEAR
} AF;

typedef enum
{
    RAND,
    HE,
    XG

} WI;
#define RELU_PARAM 0.001
#define ARRAY_LEN(x) sizeof((x)) / sizeof((x)[0])
float rand_float();
float sigmoidf(float x);
float sigmoidf_derivative(float x);
float ReLu(float x);
float ReLu_derivative(float x);
float he_init(int input_size);
float xavier_init(int input_size, int output_size);

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
    case RELU:
        return ReLu(x);
        break;
    case SIG:
        return sigmoidf(x);
        break;
    case LINEAR:
        return x;
    default:
        break;
    }
}

float dact(float x, AF af)
{

    switch (af)
    {
    case SIG:
        return sigmoidf_derivative(x);
        break;
    case RELU:
        return ReLu_derivative(x);
        break;
    case LINEAR:
        return 1;

    default:
        break;
    }
}

float he_init(int input_size)
{
    return (float)rand() / RAND_MAX * sqrt(2.0 / input_size);
}
float xavier_init(int input_size, int output_size)
{

    return (float)rand() / RAND_MAX * sqrt(1.0 / (input_size + output_size));
}

float weights_init(int input_size, int output_size, WI wi)
{

    switch (wi)
    {
    case RAND:
        return rand_float();
        break;
    case HE:
        return he_init(input_size);
        break;
    case XG:
        return xavier_init(input_size, output_size);
        /* code */
        break;

    default:
        break;
    }
}

#endif