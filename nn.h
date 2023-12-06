#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "nn_utils.h"
#include "time.h"

#ifndef NN_H
#define NN_H

typedef struct
{
    size_t count;
    size_t *arch;
    Mat *ws;  // wights matrix arrays
    Mat *gws; // gradient matrix array for waights array
    Mat *bs;  // biases matrix arrays
    Mat *gbs; // gradient matrix array for baises matrix array
    Mat *as;  // activation matrix array
    Mat *gas; // gradient matrix array for  activation matrix array

} NN;

#define NN_PRINT(nn) nn_print((nn), #nn);
#define NN_INPUT(nn) (nn_assert(nn), (nn).as[0])
#define NN_OUTPUT(nn) (nn_assert(nn), (nn).as[(nn).count])

NN nn_alloc(size_t *arch, size_t count);

void nn_assert(NN nn);
void nn_rand(NN nn);
void nn_print(NN nn, const char *name);
void nn_grad_zero(NN nn);
void nn_forward(NN nn, Mat input, AF af);
float nn_cost(NN nn, Mat input, Mat output, AF af);
void nn_backprop(NN nn, Mat input, Mat output, AF af);
void nn_train(NN nn, Mat input, Mat output, float lrate, size_t eps, AF af);
void nn_test(NN nn, Mat input, Mat output, AF af);

NN nn_alloc(size_t *arch, size_t count)
{

    assert(count > 1 && "architecture should at least contain two layer");
    NN nn;
    nn.count = count - 1;
    nn.arch = (size_t *)malloc(sizeof(size_t) * count);
    for (size_t i = 0; i < count; i++)
    {
        nn.arch[i] = arch[i];
    }

    nn.ws = (Mat *)malloc(sizeof(*nn.ws) * (nn.count));
    nn.gws = (Mat *)malloc(sizeof(*nn.gws) * (nn.count));
    assert(nn.ws != NULL && "memory not allocated for nn.ws");
    assert(nn.gws != NULL && "memory not allocated for nn.gws");

    nn.bs = (Mat *)malloc(sizeof(*nn.bs) * (nn.count));
    nn.gbs = (Mat *)malloc(sizeof(*nn.gbs) * (nn.count));
    assert(nn.bs != NULL && "memory not allocated for nn.bs");
    assert(nn.gbs != NULL && "memory not allocated for nn.gbs");

    nn.as = (Mat *)malloc(sizeof(*nn.as) * (nn.count + 1));
    nn.gas = (Mat *)malloc(sizeof(*nn.gas) * (nn.count));
    assert(nn.as != NULL && "memory not allocated for nn.as");
    assert(nn.gas != NULL && "memory not allocated for nn.gas");

    nn.as[0] = mat_alloc(1, nn.arch[0]); // input layer activations
    for (size_t i = 0; i < nn.count; i++)
    {
        nn.as[i + 1] = mat_alloc(1, nn.arch[i + 1]);
        nn.gas[i] = mat_alloc(1, nn.arch[i + 1]);
        mat_assert(nn.as[i]);
        mat_assert(nn.gas[i]);
        nn.ws[i] = mat_alloc(nn.arch[i], nn.arch[i + 1]);
        nn.gws[i] = mat_alloc(nn.arch[i], nn.arch[i + 1]);
        mat_assert(nn.ws[i]);
        mat_assert(nn.gws[i]);
        nn.bs[i] = mat_alloc(1, nn.arch[i + 1]);
        nn.gbs[i] = mat_alloc(1, nn.arch[i + 1]);
        mat_assert(nn.bs[i]);
        mat_assert(nn.gbs[i]);
    }
    srand(time(0));
    nn_rand(nn);
    return nn;
}

void nn_assert(NN nn)
{

    assert(nn.ws != NULL);
    assert(nn.gws != NULL);
    assert(nn.bs != NULL);
    assert(nn.gbs != NULL);
    assert(nn.as != NULL);
    assert(nn.gas != NULL);
}

void nn_rand(NN nn)
{
    nn_assert(nn);
    for (size_t l = 0; l < nn.count; l++)
    {
        mat_rand(nn.ws[l]);
        mat_rand(nn.bs[l]);
    }
}

void nn_print(NN nn, const char *name)
{
    nn_assert(nn);
    printf("\n %s = [\n", name);
    for (size_t l = 0; l < nn.count; l++)
    {
        MAT_PRINT(nn.as[l + 1], 1);
        MAT_PRINT(nn.ws[l], 1);
        MAT_PRINT(nn.bs[l], 1);
    }
}

void nn_grad_zero(NN nn)
{
    nn_assert(nn);
    for (size_t l = 0; l < nn.count; l++)
    {
        mat_zero(nn.gas[l]);
        mat_zero(nn.gbs[l]);
        mat_zero(nn.gws[l]);
    }
}
void nn_forward(NN nn, Mat input, AF af)
{
    nn_assert(nn);
    mat_copy(nn.as[0], input);

    for (size_t l = 0; l < nn.count; l++)
    {
        mat_dot(nn.as[l + 1], nn.as[l], nn.ws[l]);
        mat_sum(nn.as[l + 1], nn.bs[l]);
        mat_act(nn.as[l + 1], af);
    }
}
float nn_cost(NN nn, Mat input, Mat output, AF af)
{
    nn_assert(nn);
    assert(input.rows == output.rows && "input rows and ouput rows are not equal");
    assert(input.cols == NN_INPUT(nn).cols && "input cols are not equal to input layer cols");
    assert(output.cols == NN_OUTPUT(nn).cols && "output cols are not equal output layer cols");

    Mat in = mat_alloc(1, input.cols);
    float cost = 0.f;
    for (size_t tr = 0; tr < input.rows; tr++)
    {
        row_copy(in, input, tr);
        nn_forward(nn, in, af);
        float diff = 0.f;

        for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
        {
            // diff = MAT_AT(nn.as[nn.count], 0, j) - MAT_AT(output, tr, j);
            diff = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(output, tr, j);
            cost += diff * diff;
        }
    }
    cost /= input.rows;
    return cost;
}
void nn_backprop(NN nn, Mat input, Mat output, AF af)
{
    nn_assert(nn);
    assert(input.rows == output.rows && "input rows and ouput rows are not equal");
    assert(input.cols == NN_INPUT(nn).cols && "input cols are not equal to input layer cols");
    assert(output.cols == NN_OUTPUT(nn).cols && "output cols are not equal output layer cols");

    size_t n = input.rows;
    nn_grad_zero(nn);
    Mat in = mat_alloc(1, NN_INPUT(nn).cols);
    for (size_t tr = 0; tr < n; tr++)
    {
        row_copy(in, input, tr);
        nn_forward(nn, in, af);
        for (size_t l = 0; l < nn.count; l++)
        {
            mat_fill(nn.gas[l], 0);
        }
        for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
        {
            MAT_AT(nn.gas[nn.count - 1], 0, j) = 2 * (MAT_AT(nn.as[nn.count], 0, j) - MAT_AT(output, tr, j));
        }

        for (size_t l = nn.count; l > 0; l--)
        {
            for (size_t j = 0; j < nn.as[l].cols; j++)
            {
                float da = MAT_AT(nn.gas[l - 1], 0, j);
                float a = MAT_AT(nn.as[l], 0, j);
                float qa = dact(a, af);
                MAT_AT(nn.gbs[l - 1], 0, j) += da * qa;

                for (size_t k = 0; k < nn.as[l - 1].cols; k++)
                {
                    float pa = MAT_AT(nn.as[l - 1], 0, k);
                    float w = MAT_AT(nn.ws[l - 1], k, j);
                    MAT_AT(nn.gws[l - 1], k, j) += da * qa * pa;
                    if (l > 1)
                        MAT_AT(nn.gas[l - 2], 0, k) += da * qa * w;
                }
            }
        }
    }
    for (size_t l = 0; l < nn.count; l++)
    {
        for (size_t j = 0; j < nn.gws[l].rows; j++)
            for (size_t k = 0; k < nn.gws[l].cols; k++)
                MAT_AT(nn.gws[l], j, k) /= n;

        for (size_t j = 0; j < nn.gbs[l].rows; j++)
            for (size_t k = 0; k < nn.gbs[l].cols; k++)
                MAT_AT(nn.gbs[l], j, k) /= n;
    }
}

void nn_train(NN nn, Mat input, Mat output, float lrate, size_t eps, AF af)
{
    nn_assert(nn);
    assert(input.rows == output.rows && "input rows and ouput rows are not equal");
    for (size_t ep = 0; ep < eps; ep++)
    {
        nn_backprop(nn, input, output, af);
        for (size_t l = 0; l < nn.count; l++)
        {
            for (size_t j = 0; j < nn.gws[l].rows; j++)
                for (size_t k = 0; k < nn.gws[l].cols; k++)
                    MAT_AT(nn.ws[l], j, k) -= lrate * MAT_AT(nn.gws[l], j, k);

            for (size_t j = 0; j < nn.gbs[l].rows; j++)
                for (size_t k = 0; k < nn.gbs[l].cols; k++)
                    MAT_AT(nn.bs[l], j, k) -= lrate * MAT_AT(nn.gbs[l], j, k);
        }
    }
}
void nn_test(NN nn, Mat input, Mat output, AF af)
{
    nn_assert(nn);
    assert(input.rows == output.rows && "input rows and ouput rows are not equal");
    assert(input.cols == NN_INPUT(nn).cols && "input cols are not equal to input layer cols");
    assert(output.cols == NN_OUTPUT(nn).cols && "output cols are not equal output layer cols");

    Mat in = mat_alloc(1, input.cols);
    for (size_t tr = 0; tr < input.rows; tr++)
    {
        row_copy(in, input, tr);
        nn_forward(nn, in, af);

        printf("\nInput:=[");
        for (size_t j = 0; j < input.cols; j++)
        {
            printf(" %f", MAT_AT(in, 0, j));
        }
        printf("]");
        printf("Output:=[");
        for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
        {
            printf(" %f", MAT_AT(NN_OUTPUT(nn), 0, j));
        }
        printf("]");
        printf("expected:=[");
        for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
        {
            printf(" %f", MAT_AT(output, tr, j));
        }
        printf("]");
    }
}

#endif