#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "nn_utils.h"
#include "time.h"
#include <assert.h>

#ifndef NN_H
#define NN_H

#define MODEL_OUTPUT(m) (model_assert(m), (m).ls[(m).hidden_count - 1].ac)
#define MODEL_INPUT(m) (model_assert(m), (m).input)
typedef struct
{
    size_t neuron_count;
    Mat ws;
    Mat gws;
    Mat bs;
    Mat gbs;
    Mat ac;
    Mat gac;
    AF af;
    WI wi;
} Layer;

typedef struct
{
    size_t neuron_count;
    AF af;
    WI wi;
} CreateLayerInput;

Layer layer_alloc(size_t input_features, size_t neuron_count, AF af, WI wi);
typedef struct
{
    size_t hidden_count; // size of hidden layers
    Mat input;           // wights matrix arrays
    Mat output;
    Layer *ls;

} Model;
Layer layer_alloc(size_t input_features, size_t neuron_count, AF af, WI wi)
{
    assert(neuron_count > 0 && "layer muct contain at least one neuron");
    assert(input_features > 0 && "at lest one input deafture is required");
    Layer l;
    l.neuron_count = neuron_count;
    l.ws = mat_alloc(input_features, l.neuron_count);
    l.gws = mat_alloc(input_features, l.neuron_count);
    l.bs = mat_alloc(1, l.neuron_count);
    l.gbs = mat_alloc(1, l.neuron_count);
    l.ac = mat_alloc(1, l.neuron_count);
    l.gac = mat_alloc(1, l.neuron_count);
    l.af = af;
    l.wi = wi;
    return l;
}

Model create_model(size_t input_features, CreateLayerInput *arch, size_t hidden_layers_count);

void model_assert(Model m);
void model_init(Model m);
void model_print(Model m);
void model_grad_zero(Model m);
void model_forward(Model m, Mat input);
float model_cost(Model m, Mat input, Mat output);
void model_backprop(Model m, Mat input, Mat output);
void model_train(Model m, Mat input, Mat output, float lrate, size_t eps);
void model_test(Model m, Mat input, Mat output);

Model create_model(size_t input_features, CreateLayerInput *arch, size_t hidden_layers_count)
{
    assert(input_features > 0 && "at least one input featrue is requred");
    Model m;
    m.hidden_count = hidden_layers_count;
    m.input = mat_alloc(1, input_features);

    m.ls = (Layer *)malloc(sizeof(*m.ls) * m.hidden_count);
    assert(m.ls != NULL && "memory allocation failed for array  hidden layers");
    size_t ifts = input_features;
    for (size_t i = 0; i < hidden_layers_count; i++)
    {
        m.ls[i] = layer_alloc(ifts, arch[i].neuron_count, arch[i].af, arch[i].wi);
        ifts = arch[i].neuron_count;
    }
    model_init(m);
    return m;
}

void model_assert(Model m)
{
    mat_assert(m.input);
    assert(m.ls != NULL && "model not initialized");
}

void model_print(Model m)
{
    model_assert(m);
    printf("model:=[");
    for (size_t l = 0; l < m.hidden_count; l++)
    {
        printf("\n%*slayer [%d]:=[", 5, "", l);
        MAT_PRINT(m.ls[l].ws, 10);
        MAT_PRINT(m.ls[l].bs, 10);
        printf("\n%*s]", 5, "");
    }
    printf("\n]");
}

void model_init(Model m)
{
    // srand(time(0));
    srand(44);
    for (size_t l = 0; l < m.hidden_count; l++)
    {
        // mat_rand(m.ls[l].ws);
        for (int j = 0; j < m.ls[l].ws.rows; j++)
            for (int k = 0; k < m.ls[l].ws.cols; k++)
                MAT_AT(m.ls[l].ws, j, k) = weights_init(l == 0 ? m.input.cols : m.ls[l - 1].ac.cols, m.ls[l].ac.cols, m.ls[l].wi);
        // mat_rand(m.ls[l].bs);

        for (int j = 0; j < m.ls[l].bs.rows; j++)
            for (int k = 0; k < m.ls[l].bs.cols; k++)
                MAT_AT(m.ls[l].ws, j, k) = rand_float();
    }
}

void model_grad_zero(Model m)
{
    for (size_t l = 0; l < m.hidden_count; l++)
    {
        mat_fill(m.ls[l].gws, 0);
        mat_fill(m.ls[l].gbs, 0);
        mat_fill(m.ls[l].gac, 0);
    }
}
void model_forward(Model m, Mat input)
{
    model_assert(m);
    mat_copy(m.input, input);
    for (size_t l = 0; l < m.hidden_count; l++)
    {

        // if first layer then get action form input matrix else get activation form previous layer
        l == 0 ? mat_dot(m.ls[l].ac, m.input, m.ls[l].ws) : mat_dot(m.ls[l].ac, m.ls[l - 1].ac, m.ls[l].ws);
        mat_sum(m.ls[l].ac, m.ls[l].bs);
        mat_act(m.ls[l].ac, m.ls[l].af);
    }
}

float model_cost(Model m, Mat input, Mat output)
{
    model_assert(m);
    assert(input.rows == output.rows && "input rows and ouput rows are not equal");
    assert(input.cols == MODEL_INPUT(m).cols && "input cols are not equal to input layer cols");
    assert(output.cols == MODEL_OUTPUT(m).cols && "output cols are not equal output layer cols");

    Mat in = mat_alloc(1, input.cols);
    float cost = 0.f;
    for (size_t tr = 0; tr < input.rows; tr++)
    {
        row_copy(in, input, tr);
        model_forward(m, in);
        float diff = 0.f;

        for (size_t j = 0; j < MODEL_OUTPUT(m).cols; j++)
        {
            // diff = MAT_AT(nn.as[nn.count], 0, j) - MAT_AT(output, tr, j);
            diff = MAT_AT(MODEL_OUTPUT(m), 0, j) - MAT_AT(output, tr, j);
            cost += diff * diff;
        }
    }
    cost /= input.rows;
    return cost;
}

void model_backprop(Model m, Mat input, Mat output)
{
    model_assert(m);
    assert(input.rows == output.rows && "input rows and ouput rows are not equal");
    assert(input.cols == MODEL_INPUT(m).cols && "input cols are not equal to input layer cols");
    assert(output.cols == MODEL_OUTPUT(m).cols && "output cols are not equal output layer cols");

    size_t n = input.rows;
    model_grad_zero(m);
    Mat in = mat_alloc(1, MODEL_INPUT(m).cols);
    for (size_t tr = 0; tr < n; tr++)
    {
        row_copy(in, input, tr);
        model_forward(m, in);
        for (size_t l = 0; l < m.hidden_count; l++)
        {
            mat_fill(m.ls[l].gac, 0);
        }
        for (size_t j = 0; j < MODEL_OUTPUT(m).cols; j++)
        {
            MAT_AT(m.ls[m.hidden_count - 1].gac, 0, j) = 2 * (MAT_AT(m.ls[m.hidden_count - 1].ac, 0, j) - MAT_AT(output, tr, j));
        }

        for (int l = m.hidden_count - 1; l >= 0; l--)
        {
            for (size_t j = 0; j < m.ls[l].ac.cols; j++)
            {
                float da = MAT_AT(m.ls[l].gac, 0, j);
                float a = MAT_AT(m.ls[l].ac, 0, j);
                float qa = dact(a, m.ls[l].af);
                MAT_AT(m.ls[l].gbs, 0, j) += da * qa;

                int s = l == 0 ? m.input.cols : m.ls[l - 1].ac.cols;
                for (int k = 0; k < s; k++)
                {
                    float pa = l == 0 ? MAT_AT(m.input, 0, k) : MAT_AT(m.ls[l - 1].ac, 0, k);
                    float w = MAT_AT(m.ls[l].ws, k, j);
                    MAT_AT(m.ls[l].gws, k, j) += da * qa * pa;
                    if (l > 0)
                        MAT_AT(m.ls[l - 1].gac, 0, k) += da * qa * w;
                }
            }
        }
    }
    for (size_t l = 0; l < m.hidden_count; l++)
    {
        for (size_t j = 0; j < m.ls[l].gws.rows; j++)
            for (size_t k = 0; k < m.ls[l].gws.cols; k++)
                MAT_AT(m.ls[l].gws, j, k) /= n;

        for (size_t j = 0; j < m.ls[l].gbs.rows; j++)
            for (size_t k = 0; k < m.ls[l].gbs.cols; k++)
                MAT_AT(m.ls[l].gbs, j, k) /= n;
    }
}

void model_train(Model m, Mat input, Mat output, float lrate, size_t eps)
{
    model_assert(m);
    assert(input.rows == output.rows && "input rows and ouput rows are not equal");
    for (size_t ep = 0; ep < eps; ep++)
    {
        model_backprop(m, input, output);
        for (size_t l = 0; l < m.hidden_count; l++)
        {
            for (size_t j = 0; j < m.ls[l].ws.rows; j++)
                for (size_t k = 0; k < m.ls[l].ws.cols; k++)
                    MAT_AT(m.ls[l].ws, j, k) -= lrate * MAT_AT(m.ls[l].gws, j, k);

            for (size_t j = 0; j < m.ls[l].bs.rows; j++)
                for (size_t k = 0; k < m.ls[l].bs.cols; k++)
                    MAT_AT(m.ls[l].bs, j, k) -= lrate * MAT_AT(m.ls[l].gbs, j, k);
        }
    }
}
void model_test(Model m, Mat input, Mat output)
{
    model_assert(m);
    assert(input.rows == output.rows && "input rows and ouput rows are not equal");
    assert(input.cols == MODEL_INPUT(m).cols && "input cols are not equal to input layer cols");
    assert(output.cols == MODEL_OUTPUT(m).cols && "output cols are not equal output layer cols");

    Mat in = mat_alloc(1, input.cols);
    for (size_t tr = 0; tr < input.rows; tr++)
    {
        row_copy(in, input, tr);

        model_forward(m, in);

        printf("\nInput:=[");
        for (size_t j = 0; j < input.cols; j++)
        {
            printf(" %f", MAT_AT(in, 0, j));
        }
        printf("]");
        printf("Output:=[");
        for (size_t j = 0; j < MODEL_OUTPUT(m).cols; j++)
        {
            printf(" %f", MAT_AT(MODEL_OUTPUT(m), 0, j));
        }
        printf("]");
        printf("expected:=[");
        for (size_t j = 0; j < output.cols; j++)
        {
            printf(" %f", MAT_AT(output, tr, j));
        }
        printf("]");
    }
}

#endif