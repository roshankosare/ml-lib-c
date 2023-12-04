
#include "nn.h"
#include "matrix.h"

int main()
{

    float Ti[] = {
        0, 0,
        1, 0,
        0, 1,
        1, 1};

    float To[] = {
        1, 0, 0, 1, 0, 1, 1, 0};

    Mat XorTI = {
        .rows = 4,
        .cols = 2,
        .es = Ti};
    Mat XorTO = {
        .rows = 4,
        .cols = 2,
        .es = To};

    size_t arch[] = {2, 3, 2};
    size_t count = ARRAY_LEN(arch);
    NN nn = nn_alloc(arch, count);
    float lrate = 1;
    size_t eps = 100000;

    NN_PRINT(nn);

    // printf("\nCost = %f", nn_cost(nn, XorTI, XorTO, ReLU));
    // nn_train(nn, XorTI, XorTO, lrate, eps, ReLU);
    // nn_test(nn, XorTI, XorTO, ReLU);
    // printf("\nCost = %f", nn_cost(nn, XorTI, XorTO, ReLU));

    printf("\nCost = %f", nn_cost(nn, XorTI, XorTO, Sigmoid));
    nn_train(nn, XorTI, XorTO, lrate, eps, Sigmoid);
    nn_test(nn, XorTI, XorTO, Sigmoid);
    printf("\nCost = %f", nn_cost(nn, XorTI, XorTO, Sigmoid));
}