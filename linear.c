#include "nn.h"

int main()
{
    float ti[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    float to[] = {0, 2, 4, 6, 8, 10, 12, 14, 14, 18};

    Mat TI = {
        .cols = 1,
        .rows = 10,
        .es = ti};
    Mat TO = {
        .cols = 1,
        .rows = 10,
        .es = to};

    CreateLayerInput arch[] = {{.neuron_count = 1, .af = LINEAR, .wi = RAND}};
    size_t count = ARRAY_LEN(arch);
    Model m = create_model(1, arch, count);
    float rate = 1e-2;
    size_t ep = 1000;

    model_train(m, TI, TO, rate, ep);
    model_test(m, TI, TO);
}
