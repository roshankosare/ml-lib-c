
#include "nnt.h"
#include "matrix.h"

int main()
{

    float Ti[] = {
        0, 0,
        1, 0,
        0, 1,
        1, 1};

    float To[] = {
        0, 1, 1, 1};

    Mat XorTI = {
        .rows = 4,
        .cols = 2,
        .es = Ti};
    Mat XorTO = {
        .rows = 4,
        .cols = 1,
        .es = To};

    CreateLayerInput arch[] = {{.neuron_count = 2, af : Sigmoid}, {.neuron_count = 1, af : Sigmoid}};
    size_t count = ARRAY_LEN(arch);
    Model m = create_model(2, arch, count);
    model_rand(m);
    float lrate = 1;
    size_t eps = 100000;
    printf("cost:%f", model_cost(m, XorTI, XorTO));
    model_train(m, XorTI, XorTO, lrate, eps);
    model_test(m, XorTI, XorTO);
    printf("cost:%f", model_cost(m, XorTI, XorTO));
}