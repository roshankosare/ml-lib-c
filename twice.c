
#include "nn.h"
int main()
{

    float ti[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    float to[] = {0, 2, 4, 6, 10, 12, 14, 16, 18};

    Mat TI = {
        .cols = 1,
        .rows = 5,
        .es = ti};
    Mat TO = {
        .cols = 1,
        .rows = 5,
        .es = to};

    size_t arch[] = {1, 1};
    size_t count = ARRAY_LEN(arch);
    NN nn = nn_alloc(arch, count);
    float rate = 1e-1;
    size_t ep = 5000;

    nn_train(nn, TI, TO, rate, ep, ReLU);
    nn_test(nn, TI, TO, ReLU);
    //  nn_train(nn, TI, TO, rate, ep, Sigmoid);
    // nn_test(nn, TI, TO, Sigmoid);

}