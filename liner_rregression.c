#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "matrix.h"
#include "nn.h"

int main()
{

    Mat m;
    m = csvToMat("train.csv");
    Mat trainInput = mat_sub(m, 2, 1, 0);
    Mat trainOutput = mat_sub(m, 2, 1, 1);
    m = csvToMat("test.csv");
    Mat testInput = mat_sub(m, 20, 1, 0);
    Mat testnOutput = mat_sub(m, 20, 1, 1);

    CreateLayerInput arch[] = {{.neuron_count = 1, .af = LINEAR, .wi = RAND}};
    size_t count = ARRAY_LEN(arch);
    Model model = create_model(1, arch, count);
    float rate = 1e-2;
    size_t ep = 1;

    model_train(model, trainInput, trainOutput, rate, ep);
    model_test(model, testInput, testnOutput);
}
