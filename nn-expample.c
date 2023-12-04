
#include "nn.h"

int main(){

    size_t arch[] = {2,2,1};
    size_t count = ARRAY_LEN(arch);

    NN nn = nn_alloc(arch,count);
    nn_rand(nn);
    NN_PRINT(nn);
}