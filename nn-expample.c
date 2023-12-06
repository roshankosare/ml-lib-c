
#include "nnt.h"

int main()
{

    CreateLayerInput arch[] = {{.neuron_count = 2, af : ReLU}, {.neuron_count = 1, af : Sigmoid}};
    size_t count = ARRAY_LEN(arch);
    Model m = create_model(2, arch, count);
    model_rand(m);
    // // model_print(m);
    // MAT_PRINT(MODEL_INPUT(m), 1);
}